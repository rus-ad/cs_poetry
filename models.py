from transformers import GPT2LMHeadModel, AutoTokenizer


class BaseModel:

    def __init__(self, backbone_id):
        self.backbone_id = backbone_id

    def load_model(self):
        return GPT2LMHeadModel.from_pretrained(self.backbone_id)

    def load_tokenizer(self):
        return AutoTokenizer.from_pretrained(
            self.backbone_id,
            pad_token="<pad>",
            eos_token="<pad>",
        )

    def get_models(self):
        return {
            'model': self.load_model(),
            'tokenizer': self.load_tokenizer(),
        }
