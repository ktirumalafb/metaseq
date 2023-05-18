# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch



from metaseq import metrics, utils
from metaseq.criterions import BaseCriterion, register_criterion

import logging

import json
import os

import time

import os

logger = logging.getLogger(__name__)


try:
    from megatron.mpu.cross_entropy import (
        vocab_parallel_cross_entropy,
    )

    has_megatron_submodule = True
except (ImportError, ModuleNotFoundError):
    has_megatron_submodule = False


@register_criterion("vocab_parallel_cross_entropy")
class VocabParallelCrossEntropyCriterion(BaseCriterion):
    def __init__(self, task):
        super().__init__(task)
        if not has_megatron_submodule:
            raise ImportError(
                "\n\nPlease install megatron using the setup instructions!"
            )

    def forward(self, model, sample, reduce=True, compute_metrics=False):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        target = sample["target"]
        
        has_pad = target.eq(self.padding_idx).any().item()

        net_output = model(**sample["net_input"])


        loss = vocab_parallel_cross_entropy(net_output[0].float(), target)
        loss_original_arr = loss.detach().clone()
        if has_pad:
            loss = loss * (target != self.padding_idx)


        loss = loss.sum()

        # When using target loss only, use num tokens in target only as the sample_size
        # See StreamingSrcTgtDataset
        sample_size = (
            sample["ntokens_target"]
            if "ntokens_target" in sample
            else sample["ntokens"]
        )
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
            logging_output["targets_for_output"] = target
            logging_output["loss_for_output"] = loss_original_arr
            logging_output["path_infos_for_output"] = sample["path_infos"]
            logging_output["id_for_output"] = sample["id"]

            # for ssl prototypes
            logging_output["final_embedding"] = actv.transpose(0,1)


        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs, data_pruning_metrics=None, data_pruning_metrics_savedir=None, length_dataset=None, final_folder_name=None) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        for type_ in ("actv", "pos", "tok", "emb"):
            key = f"{type_}_norm"
            if any(key in log for log in logging_outputs):
                actv_norm = sum(log.get(key, 0) for log in logging_outputs)
                metrics.log_scalar(key, actv_norm / ntokens, round=3)

        if any("ndocseps" in log for log in logging_outputs):
            # nsentences = batch size
            nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
            # ndocseps = number of document separators we found
            ndocseps = sum(log.get("ndocseps", 0) for log in logging_outputs)
            # so docs/example = (1 + ndocseps) / example = (ndocseps + nsents) / nsents
            metrics.log_scalar("docsperex", (ndocseps + nsentences) / nsentences)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
        )
        # rounding to nearest int
        def serialize_tensor(t):
            return [int(x) for x in t.cpu().detach().numpy().tolist()]

        # rounding to int + 5 decimal digits
        def serialize_tensor_ppl(t):
            return [int(utils.get_perplexity(x) * 10000) for x in t.cpu().detach().numpy().tolist()]

        # not rounding at all
        def serialize_tensor_to_numpy(t):
            return t.cpu().detach().numpy()
        
        data_pruning_metrics_savedir = os.path.join(data_pruning_metrics_savedir, final_folder_name)

        if data_pruning_metrics:
            if "ppl" in data_pruning_metrics:
                # print("-------------- Starting printing here ------------------------")
                # print(f"len of logging outputs: {len(logging_outputs)}")
                with open(f"{data_pruning_metrics_savedir}/ppl_output.json", "a") as f:
                    for logging_output in logging_outputs:
                        batch_size = logging_output["targets_for_output"].shape[0]

                        # print(f"batch size: {batch_size}")

                        size_id_tensor = len(logging_output["id_for_output"])
                        # print(f"size of id tensor: {size_id_tensor}")

                        size_path_infos = len(logging_output["path_infos_for_output"])
                        # print(f"size of path info list: {size_path_infos}")

                        size_targets_tensor = logging_output["targets_for_output"].shape[0]
                        # print(f"size targets: {size_targets_tensor}")

                        size_loss_for_output = logging_output["loss_for_output"].shape[0]
                        # print(f"size loss for output: {size_loss_for_output}")

                        # print("******* Going through now ************")

                        
                        for i in range(batch_size):
                           

                            num_pad = sum(int(x==1) for x in serialize_tensor(logging_output["targets_for_output"][i]))
                            length_final = len(serialize_tensor(logging_output["loss_for_output"][i]))

                            # print(f"num pad: {num_pad}")
                            # print(f"length_final: {length_final}")

                            log_line = {
                                "id": int(logging_output["id_for_output"][i]) if i < size_id_tensor else None,
                                "t":  serialize_tensor(logging_output["targets_for_output"][i]),
                                "p":  serialize_tensor_ppl(logging_output["loss_for_output"][i]),
                                "path_info": logging_output["path_infos_for_output"][i] if i < size_path_infos else None,
                                "num_pad": num_pad,
                                "length_final": length_final
                            }
                            f.write(json.dumps(log_line) + "\n")

                        # print("*********************************************")
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
                        hash_targets = hash(logging_output["targets_for_output"])
                        hash_path_infos = hash("".join(logging_output["path_infos_for_output"]))

                        hash_to_add = str(hash_targets) + "" + str(hash_path_infos)
                        batch_size = logging_output["targets_for_output"].shape[0]

                        num_adding = logging_output["final_embedding"].shape[0]
                        assert batch_size == num_adding

                        # Write the embedding
                        # embedding_out_file[counter: counter+num_adding] = serialize_tensor_to_numpy(logging_output["final_embedding"])

                        # Write the metadata
                        for i in range(batch_size):
                            num_pad = sum(int(x==1) for x in serialize_tensor(logging_output["targets_for_output"][i]))
                            length_final = len(serialize_tensor(logging_output["loss_for_output"][i])) 
                            
                            # If the last 3 tokens were pad, then we want arr[-4] to get the lost non-pad token embedding
                            sub_from_end = -1*num_pad - 1
                            embedding_out_file[counter+i] = serialize_tensor_to_numpy(logging_output["final_embedding"][i, sub_from_end])
       
                            log_line = {
                                "index_in_arr":  counter + i,
                                "path_info": logging_output["path_infos_for_output"][i],
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



    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improve distributed training speed.
        """
        return False
