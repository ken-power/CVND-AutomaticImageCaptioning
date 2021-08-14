import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super().__init__()

        resnet = models.resnet50(pretrained=True)

        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)

        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()

        self.hidden_size = hidden_size

        # embedding layer that turns words into a vector of a specified size
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)

        # The LSTM takes embedded vectors as inputs and outputs hidden states of hidden_size
        self.lstm = nn.LSTM(input_size=embed_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)

        # the linear layer that maps the hidden state output dimension
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        captions = captions[:, :-1]
        embeds = self.word_embeddings(captions)

        # Concatenating features to embedding
        # torch.cat 3D tensors
        inputs = torch.cat((features.unsqueeze(1), embeds), 1)

        lstm_out, hidden = self.lstm(inputs)
        outputs = self.linear(lstm_out)

        return outputs

    def sample(self, inputs, states=None, max_len=20):
        """
        Accepts pre-processed image tensor (inputs) and returns predicted sentence

        :param inputs: pre-processed image tensor
        :return: predicted sentence (list of tensor ids of length max_len)
        """
        predicted_sentence = []

        for i in range(max_len):
            lstm_out, states = self.lstm(inputs, states)

            lstm_out = lstm_out.squeeze(1)
            lstm_out = lstm_out.squeeze(1)
            outputs = self.linear(lstm_out)

            # Get maximum probabilities
            target = outputs.max(1)[1]

            # Append result into predicted_sentence list
            predicted_sentence.append(target.item())

            # Update the input for next iteration
            inputs = self.word_embeddings(target).unsqueeze(1)

        return predicted_sentence
