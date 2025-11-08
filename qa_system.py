# Mini Question-Answering System using Pretrained Transformer (BERT)

from transformers import pipeline

# Load pretrained QA model
qa_pipeline = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

# Health domain context – Nutrition and Healthy Lifestyle
context = """
Nutrition plays a vital role in maintaining good health and preventing chronic diseases. A balanced diet provides the body with 
essential nutrients such as carbohydrates, proteins, fats, vitamins, and minerals that are necessary for proper functioning. Poor 
dietary habits, such as consuming excessive processed foods, sugar, and unhealthy fats, can lead to obesity, diabetes, heart disease, 
and other lifestyle disorders. In contrast, a diet rich in fruits, vegetables, whole grains, lean proteins, and healthy fats supports 
physical and mental well-being. Hydration is equally important, as water helps regulate body temperature, transport nutrients, and 
remove toxins.

Regular physical activity complements good nutrition. Engaging in at least 30 minutes of moderate exercise daily improves cardiovascular 
health, boosts metabolism, and strengthens muscles and bones. Exercise also releases endorphins, which improve mood and reduce stress. 
Adequate sleep is another key factor in maintaining a healthy lifestyle. Lack of sleep can impair concentration, weaken the immune system, 
and increase the risk of chronic diseases.

In recent years, technology has significantly influenced health and nutrition. Fitness trackers, smartwatches, and mobile apps allow 
individuals to monitor calorie intake, heart rate, and daily activity levels. Artificial intelligence is being used to design personalized 
diet plans based on an individual’s age, body type, and health conditions. Nutrition education through digital platforms has also increased 
awareness about mindful eating and portion control.

Despite these advancements, unhealthy eating patterns remain common due to fast food culture, lack of time, and misinformation. Public 
health initiatives must focus on promoting healthy habits from an early age, especially in schools and workplaces. Achieving long-term 
wellness requires a holistic approach that balances nutrition, exercise, mental well-being, and rest. By making informed choices and using 
modern tools wisely, individuals can take control of their health and lead more productive, happier lives.
"""

# Ask questions until user exits
while True:
    question = input("\nAsk a question about nutrition (or type 'exit' to quit): ")
    if question.lower() == "exit":
        break
    result = qa_pipeline(question=question, context=context)
    print(f"Answer: {result['answer']} (Confidence: {result['score']:.2f})")
