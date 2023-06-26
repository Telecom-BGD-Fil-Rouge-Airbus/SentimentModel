from inference_sentiment_ternary import load_model, predict_ternary_sentiment, SentimentClassifier


model_path = "./ternary_sentiment_model.pth"

model = load_model(model_path)


texts_list = ["I'm not capable of doing it.", "I believe in myself.", "I don't see the point in trying.", "I'm excited to give it a shot.", "I'm not good enough.", "I'm capable of handling this.", "This is a disaster waiting to happen.", "This is a great opportunity to learn.", "I'm always so unlucky.", "I'm grateful for this experience.", "No pain, no gain.", "It's not you, it's me.", "It could have been worse.", "I'm not as young as I used to be.", "You're too good to be true.", "The weather today is partly cloudy.", "I ate a turkey sandwich for lunch.", "The book I'm reading is about history.", "My favorite color is blue.", "I need to go grocery shopping this weekend."]

for text in texts_list:

    prediction = predict_ternary_sentiment(model, text)

    print("text :",text)

    if prediction == 0:
        print("The sentiment is negative")
    elif prediction == 1:
        print("The sentiment is neutral")
    elif prediction == 2:
        print("The sentiment is positive")