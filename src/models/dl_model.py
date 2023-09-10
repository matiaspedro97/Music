from transformers import AutoModelForAudioClassification


class AudioClassifierHead:
    def __init__(self, model_id: str) -> None:
        # Model ID 
        self.model_id = model_id

    def load_model(self, label2id: dict):
        # ID to Label
        id2label = {v: k for k, v in label2id.items()}

        # Number of labels
        n_labels = len(label2id)

        # instantiate pre-trained model
        model = AutoModelForAudioClassification.from_pretrained(
            self.model_id,
            num_labels=n_labels,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True
        )

        return model

