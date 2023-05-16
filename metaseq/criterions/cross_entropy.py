# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F

from metaseq import metrics, utils
from metaseq.criterions import BaseCriterion, register_criterion


def nll_loss(lprobs, target, ignore_index=None, reduction="mean"):
    """Like torch.nn.functional.nll_loss but works for large inputs."""
    if lprobs.numel() < 2e9:
        return F.nll_loss(
            lprobs, target, ignore_index=ignore_index, reduction=reduction
        )
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
    if reduction == "mean":
        nll_loss = nll_loss.mean()
    elif reduction == "sum":
        nll_loss = nll_loss.sum()
    elif reduction == "none":
        pass
    else:
        raise NotImplementedError
    return nll_loss


@register_criterion("cross_entropy")
class CrossEntropyCriterion(BaseCriterion):
    def __init__(self, task):
        super().__init__(task)

    def forward(self, model, sample, reduce=True, compute_metrics=False):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        from metaseq import pdb; pdb.set_trace()
        net_output = model(**sample["net_input"])
        loss, _ = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample["ntokens"]
        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if "src_tokens" in sample["net_input"] and hasattr(self.task, "eod"):
            logging_output["ndocseps"] = (sample["target"] == self.task.eod).sum()
        if (
            len(net_output) >= 2
            and isinstance(net_output[1], dict)
            and "inner_states" in net_output[1]
        ):
            with torch.no_grad():
                # yank out the inner states we wish to instrument
                # see transformer_decoder.py ModelParallelTransformerDecoder.extract_features
                emb, *_, actv = net_output[1]["inner_states"]
                assert isinstance(
                    emb, dict
                ), "Expecting the first inner state to be a dict of embedding representations"
                emb["actv"] = actv  # throw on final for code brevity
                for key, value in emb.items():
                    if value is None:
                        # maybe future proofing relative positional embeddings
                        continue
                    value = emb[key]
                    logging_output[f"{key}_norm"] = value.norm(p=2, dim=-1).sum(
                        dtype=torch.float32
                    )
        if compute_metrics:
            logging_output["targets"] = targets
            logging_output["log_probs"] = target_log_probs
            
            logging_output["path_infos"] = sample["path_infos"]

            # for ssl prototypes
            logging_output["final_embedding"] = actv.transpose(0,1)

            # compute el2n score
            nsentences = sample["target"].size(0)
            loss_vector, _ = self.compute_loss(model, net_output, sample, reduce=False)
            loss_vector = loss_vector.reshape((nsentences,-1))

            # 2 norm of loss vector
            logging_output["el2n_score"] = torch.linalg.norm(loss_vector, dim=1)
            logging_output["id"] = sample["id"]

        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output[0], log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample).view(-1)
        loss = nll_loss(
            lprobs,
            target,
            ignore_index=self.padding_idx,
            reduction="sum" if reduce else "none",
        )
        return loss, loss

    @staticmethod
    def reduce_metrics(logging_outputs, data_pruning_metrics=None, data_pruning_metrics_savedir=None, length_dataset=None) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        for type_ in ("actv", "pos", "tok", "emb"):
            key = f"{type_}_norm"
            if any(key in log for log in logging_outputs):
                actv_norm = sum(log.get(key, 0) for log in logging_outputs)
                metrics.log_scalar(key, actv_norm / ntokens, round=3)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
            )
        else:
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
            )
        # rounding to nearest int
        def serialize_tensor(t):
            return [int(x) for x in t.cpu().detach().numpy().tolist()]

        # rounding to int + 5 decimal digits
        def serialize_tensor_ppl(t):
            return [int(x * 10000) for x in t.cpu().detach().numpy().tolist()]

        # not rounding at all
        def serialize_tensor_to_numpy(t):
            return t.cpu().detach().numpy()

        if data_pruning_metrics:
            if "ppl" in data_pruning_metrics:
                with open(f"{data_pruning_metrics_savedir}/ppl_output.json", "a") as f:
                    for logging_output in logging_outputs:
                        batch_size = logging_output["targets"].shape[0]
                        for i in range(batch_size):

                            num_pad = sum(int(x==1) for x in serialize_tensor(logging_output["targets"][i]))
                            length_final = len(serialize_tensor(logging_output["log_probs"][i]))

                            log_line = {
                                "id": int(logging_output["id"][i]),
                                "t":  serialize_tensor(logging_output["targets"][i]),
                                "p":  serialize_tensor_ppl(logging_output["log_probs"][i]),
                                "path_info": logging_output["path_infos"][i],
                                "num_pad": num_pad,
                                "length_final": length_final
                            }
                            f.write(json.dumps(log_line) + "\n")
                logger.info("Done writing ppl info!")


            if "ssl_prototypes_compute_centroids" in data_pruning_metrics:


                if not os.path.isdir(f"{data_pruning_metrics_savedir}/ssl_embeddings"):
                    os.mkdir(f"{data_pruning_metrics_savedir}/ssl_embeddings")

                if not os.path.isfile(f"{data_pruning_metrics_savedir}/ssl_embeddings/counter.txt"):
                    with open(f"{data_pruning_metrics_savedir}/ssl_embeddings/counter.txt", "w") as f_counter:
                        f_counter.write("0")

                # counter for where you are in the memmap
                counter = int(open(f"{data_pruning_metrics_savedir}/ssl_embeddings/counter.txt", "r").readlines()[0])

                # guess the model embedding size by looking at first output for the batch
                # model_embedding_size = logging_outputs[0]["final_embedding"].shape[1]

                model_embedding_size = logging_outputs[0]["final_embedding"].shape[-1]

                if not os.path.isfile(f"{data_pruning_metrics_savedir}/ssl_embeddings/embedding.npy"):
                    # Create the memmap file in w+ mode
                    embedding_out_file = np.memmap(f"{data_pruning_metrics_savedir}/ssl_embeddings/embedding.npy", dtype='float32', mode='w+', shape=(length_dataset,model_embedding_size))
                else:
                    # Append to existing file
                    embedding_out_file = np.memmap(f"{data_pruning_metrics_savedir}/ssl_embeddings/embedding.npy", dtype='float32', mode='r+', shape=(length_dataset,model_embedding_size))

                
                with open(f"{data_pruning_metrics_savedir}/ssl_embeddings/index.json", "a") as f_index_out:

                    for logging_output in logging_outputs:
                        hash_targets = hash(logging_output["targets"])
                        hash_path_infos = hash("".join(logging_output["path_infos"]))

                        hash_to_add = str(hash_targets) + "" + str(hash_path_infos)
                        batch_size = logging_output["targets"].shape[0]

                        num_adding = logging_output["final_embedding"].shape[0]
                        assert batch_size == num_adding

                        # Write the embedding
                        # embedding_out_file[counter: counter+num_adding] = serialize_tensor_to_numpy(logging_output["final_embedding"])

                        # Write the metadata
                        for i in range(batch_size):
                            num_pad = sum(int(x==1) for x in serialize_tensor(logging_output["targets"][i]))
                            length_final = len(serialize_tensor(logging_output["log_probs"][i])) 
                            
                            # If the last 3 tokens were pad, then we want arr[-4] to get the lost non-pad token embedding
                            sub_from_end = -1*num_pad - 1
                            embedding_out_file[counter+i] = serialize_tensor_to_numpy(logging_output["final_embedding"][i, sub_from_end])
       
                            log_line = {
                                "index_in_arr":  counter + i,
                                "path_info": logging_output["path_infos"][i],
                                "length": length_final,
                                "num_pad": num_pad
                            }
                            f_index_out.write(json.dumps(log_line) + "\n")

                        # Update the counter
                        counter += num_adding



                    logger.info("Done writing embedding info!")
                
                # After we are done, update the counter in the file
                with open(f"{data_pruning_metrics_savedir}/ssl_embeddings/counter.txt", "w") as f_counter:
                    f_counter.write(str(counter))
  

            if "el2n" in data_pruning_metrics:
                with open(f"{data_pruning_metrics_savedir}/el2n.json", "a") as f:
                    for logging_output in logging_outputs:
                        batch_size = logging_output["targets"].shape[0]

                        
                        for i in range(batch_size):

                            num_pad = sum(int(x==1) for x in serialize_tensor(logging_output["targets"][i]))
                            length_final = len(serialize_tensor(logging_output["log_probs"][i]))
                            log_line = {
                                "path_info": logging_output["path_infos"][i],
                                "el2n_metric": logging_output["el2n_score"][i].item(),
                                "length": length_final,
                                "num_pad": num_pad
                            }
                            f.write(json.dumps(log_line) + "\n")
                logger.info("Done writing el2n info!")

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improve distributed training speed.
        """
        return True
