import google.generativeai as genai
genai.configure(api_key='AIzaSyBF6bKuEG9LhwPM7X_LaDdUipFvjw56GYc')
model = genai.GenerativeModel('gemini-1.5-flash')
response = model.generate_content("whats 1+1")
print(response.text)




