import csv
from tkinter import messagebox

def read_csv(filename):
    data = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row)
    return data

def write_csv(filename, data):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

def insert_data(app):
    sentiment = app.input_sentiment.get()
    hashtags = app.input_hashtags.get()
    retweets = app.input_retweets.get()
    likes = app.input_likes.get()
    month = app.input_month.get()
    hour = app.input_hour.get()

    data = [app.input_text.get(), sentiment, hashtags, retweets, likes, month, hour]
    output_file_path = 'Sentiment_dataset/sentimentdataset.csv'

    with open(output_file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(data)

    messagebox.showinfo("Success", "Data inserted successfully.")

def check_inputs(app):
    if all([
        app.input_text.get(),
        app.input_sentiment.get(),
        app.input_hashtags.get(),
        app.input_retweets.get(),
        app.input_likes.get(),
        app.input_month.get(),
        app.input_hour.get()
    ]):
        messagebox.showinfo("Check", "All fields are filled!")
    else:
        messagebox.showwarning("Check", "Please fill all fields!")
