import logging
import os
import sys
import warnings
from dataclasses import dataclass, field, asdict
from typing import Optional, List

import datasets
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import DatasetDict, load_dataset

import transformers
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import send_example_telemetry
from transformers.utils.versions import require_version

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

require_version(
    "datasets>=1.14.0",
    "To fix: pip install -r examples/pytorch/audio-classification/requirements.txt",
)


def list_field(default=None, metadata=None):
    return field(default_factory=lambda: default, metadata=metadata)


@dataclass
class DistillationTrainingArguments:
    """
    Arguments pertaining to distillation settings.
    """

    alpha: float = field(
        default=0.5,
        metadata={
            "help": "Hyperparameter to control the relative strength of each loss."
        },
    )
    temperature: float = field(
        default=2.0,
        metadata={"help": "Scale factor of logits to soften the probabilities."},
    )
    layer_prefix: str = field(
        default=None,
        metadata={
            "help": "Layer name prefix to copy from teacher model. E.g. `wav2vec2.encoder.layers`."
        },
    )
    delimiter: str = field(
        default=".", metadata={"help": "Layer name components delimiter."}
    )
    teacher_blocks: List[str] = list_field(
        default=None,
        metadata={
            "help": "A list of teacher block indices to copy from. E.g. `'0 2 4 6 8 10'`"
        },
    )


class MultiLabelDistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs_stu = model(**inputs)
        logits_stu = outputs_stu.logits
        bce_loss_fct = torch.nn.BCEWithLogitsLoss()
        loss_bce = bce_loss_fct(
            logits_stu.view(-1, self.model.config.num_labels),
            labels.float().view(-1, self.model.config.num_labels),
        )
        with torch.no_grad():
            outputs_tea = self.teacher_model(**inputs)
            logits_tea = outputs_tea.logits
        kd_loss_fct = nn.KLDivLoss(reduction="batchmean")
        loss_kd = self.args.temperature**2 * kd_loss_fct(
            F.log_softmax(logits_stu / self.args.temperature, dim=-1),
            F.softmax(logits_tea / self.args.temperature, dim=-1),
        )
        loss = self.args.alpha * loss_bce + (1.0 - self.args.alpha) * loss_kd
        return (loss, outputs_stu) if return_outputs else loss


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "Name of a dataset from the datasets package"}
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "A file containing the training audio paths and labels."},
    )
    eval_file: Optional[str] = field(
        default=None,
        metadata={"help": "A file containing the validation audio paths and labels."},
    )
    train_split_name: str = field(
        default="train",
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"
        },
    )
    eval_split_name: str = field(
        default="validation",
        metadata={
            "help": (
                "The name of the training data set split to use (via the datasets library). Defaults to 'validation'"
            )
        },
    )
    audio_column_name: str = field(
        default="audio",
        metadata={
            "help": "The name of the dataset column containing the audio data. Defaults to 'audio'"
        },
    )
    label_column_name: Optional[str] = field(
        default="label",
        metadata={
            "help": "The name of the dataset column containing the labels. Defaults to 'label'"
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_length_seconds: float = field(
        default=20,
        metadata={
            "help": "Audio clips will be randomly cut to this length during training if the value is set."
        },
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="facebook/wav2vec2-base",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from the Hub"
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    feature_extractor_name: Optional[str] = field(
        default=None, metadata={"help": "Name or path of preprocessor config."}
    )
    freeze_feature_encoder: bool = field(
        default=True,
        metadata={"help": "Whether to freeze the feature encoder layers of the model."},
    )
    attention_mask: bool = field(
        default=True,
        metadata={
            "help": "Whether to generate an attention mask in the feature extractor."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    freeze_feature_extractor: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to freeze the feature extractor layers of the model."
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={
            "help": "Will enable to load a pretrained model whose head dimensions are different."
        },
    )

    def __post_init__(self):
        if not self.freeze_feature_extractor and self.freeze_feature_encoder:
            warnings.warn(
                "The argument `--freeze_feature_extractor` is deprecated and "
                "will be removed in a future version. Use `--freeze_feature_encoder`"
                "instead. Setting `freeze_feature_encoder==True`.",
                FutureWarning,
            )
        if self.freeze_feature_extractor and not self.freeze_feature_encoder:
            raise ValueError(
                "The argument `--freeze_feature_extractor` is deprecated and "
                "should not be used in combination with `--freeze_feature_encoder`."
                "Only make use of `--freeze_feature_encoder`."
            )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (
            ModelArguments,
            DataTrainingArguments,
            TrainingArguments,
            DistillationTrainingArguments,
        )
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, distil_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        (
            model_args,
            data_args,
            training_args,
            distil_args,
        ) = parser.parse_args_into_dataclasses()

    # copy alpha and temperature values from DistillationTrainingArguments to TrainingArguments
    for key, value in asdict(distil_args).items():
        setattr(training_args, key, value)

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_audio_classification", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu} "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to train from scratch."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Initialize our dataset and prepare it for the audio classification task.
    raw_datasets = DatasetDict()
    raw_datasets["train"] = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        split=data_args.train_split_name,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    raw_datasets["eval"] = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        split=data_args.eval_split_name,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    if data_args.audio_column_name not in raw_datasets["train"].column_names:
        raise ValueError(
            f"--audio_column_name {data_args.audio_column_name} not found in dataset '{data_args.dataset_name}'. "
            "Make sure to set `--audio_column_name` to the correct audio column - one of "
            f"{', '.join(raw_datasets['train'].column_names)}."
        )

    # Setting `return_attention_mask=True` is the way to get a correctly masked mean-pooling over
    # transformer outputs in the classifier, but it doesn't always lead to better accuracy
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_args.feature_extractor_name or model_args.model_name_or_path,
        return_attention_mask=model_args.attention_mask,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # `datasets` takes care of automatically loading and resampling the audio,
    # so we just need to set the correct target sampling rate.
    raw_datasets = raw_datasets.cast_column(
        data_args.audio_column_name,
        datasets.features.Audio(sampling_rate=feature_extractor.sampling_rate),
    )

    model_input_name = feature_extractor.model_input_names[0]

    def preprocess_data(examples):
        # get audio arrays
        audio_arrays = [x["array"] for x in examples[data_args.audio_column_name]]
        # encode batch of audio
        inputs = feature_extractor(
            audio_arrays, sampling_rate=feature_extractor.sampling_rate
        )
        # add labels
        labels_batch = {k: examples[k] for k in examples.keys() if k in labels}
        # create numpy array of shape (batch_size, num_labels)
        labels_matrix = np.zeros((len(audio_arrays), len(labels)))
        # fill numpy array
        for idx, label in enumerate(labels):
            labels_matrix[:, idx] = labels_batch[label]

        output_batch = {model_input_name: inputs.get(model_input_name)}
        output_batch["labels"] = labels_matrix.tolist()

        return output_batch

    def multi_label_metrics(predictions, labels, threshold=0.5):
        # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(predictions)).cpu().numpy()
        # next, use threshold to turn them into integer predictions
        y_pred = np.zeros(probs.shape)
        y_pred[np.where(probs >= threshold)] = 1
        # finally, compute metrics
        f1_micro_average = f1_score(y_true=labels, y_pred=y_pred, average="micro")
        roc_auc = roc_auc_score(labels, y_pred, average="micro")
        accuracy = accuracy_score(labels, y_pred)
        mAP = average_precision_score(labels, probs, average="micro")
        # return as dictionary
        metrics = {
            "f1": f1_micro_average,
            "roc_auc": roc_auc,
            "accuracy": accuracy,
            "mAP": mAP,
        }
        return metrics

    def compute_metrics(p: EvalPrediction):
        """Computes mean average precision (mAP) score"""
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        result = multi_label_metrics(predictions=preds, labels=p.label_ids)
        return result

    teacher_config = AutoConfig.from_pretrained(
        model_args.config_name or model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    teacher_model = AutoModelForAudioClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=teacher_config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    ).to(training_args.device)

    labels = list(teacher_config.id2label.values())

    layer_num_idx: int = len(distil_args.layer_prefix.split(distil_args.delimiter))
    num_hidden_layers: int = len(distil_args.teacher_blocks)
    assert num_hidden_layers <= teacher_model.config.num_hidden_layers

    student_config = AutoConfig.from_pretrained(
        model_args.config_name or model_args.model_name_or_path,
        num_hidden_layers=num_hidden_layers,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    student_model = AutoModelForAudioClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=student_config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )

    # initialize student's weights from teacher's
    teacher_weights = teacher_model.state_dict()
    student_weights = student_model.state_dict()

    for name, param in student_weights.items():
        if name.startswith(distil_args.layer_prefix):
            # split layer name to its components
            student_layer_name_comps = name.split(distil_args.delimiter)
            student_layer_num = student_layer_name_comps[layer_num_idx]
            # replace the layer num with teacher's layer num
            student_layer_name_comps[layer_num_idx] = distil_args.teacher_blocks[
                int(student_layer_num)
            ]
            # join to get teacher's layer name
            teacher_layer_name = distil_args.delimiter.join(student_layer_name_comps)
            # in-place copy to student params
            param.copy_(teacher_weights[teacher_layer_name])

    # freeze the convolutional waveform encoder
    if model_args.freeze_feature_encoder:
        student_model.freeze_feature_encoder()

    if training_args.do_train:
        if data_args.max_train_samples is not None:
            raw_datasets["train"] = (
                raw_datasets["train"]
                .shuffle(seed=training_args.seed)
                .select(range(data_args.max_train_samples))
            )
        # Set the training transforms
        raw_datasets["train"].set_transform(preprocess_data, output_all_columns=False)

    if training_args.do_eval:
        if data_args.max_eval_samples is not None:
            raw_datasets["eval"] = (
                raw_datasets["eval"]
                .shuffle(seed=training_args.seed)
                .select(range(data_args.max_eval_samples))
            )
        # Set the validation transforms
        raw_datasets["eval"].set_transform(preprocess_data, output_all_columns=False)

    # Initialize our trainer
    trainer = MultiLabelDistillationTrainer(
        model=student_model,
        teacher_model=teacher_model,
        args=training_args,
        train_dataset=raw_datasets["train"] if training_args.do_train else None,
        eval_dataset=raw_datasets["eval"] if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=feature_extractor,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Write model card and (optionally) push to hub
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "audio-classification",
        "dataset": data_args.dataset_name,
        "tags": ["audio-classification"],
    }
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()
