import torch

# Build the Sentiment Classifier class 
class SentimentClassifier(torch.nn.Module):
    
    # Constructor class 
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
    
    # Forward propagaion class
    def forward(self, input_ids, attention_mask, return_dict=False):
        _, pooled_output = self.bert(
          input_ids=input_ids,
          attention_mask=attention_mask, 
          return_dict = return_dict
        )
        #  Add a dropout layer 
        output = self.drop(pooled_output)
        return self.out(output)