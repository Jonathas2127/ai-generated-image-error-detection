"""Historical reference: retired API models; see legacy/README.md."""
import os
import json
import base64
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from groq import Groq
import re  # Import regular expressions for category extraction

# Initialize the Groq API client
api_key = os.environ["GROQ_API_KEY"]
client = Groq(api_key=api_key)

# Function to encode the image in base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Function to classify images using Groq Vision API
def classify_image_with_groq_base64(encoded_image, filename, ground_truth_labels):
    prompt_template = """
    Classify the following image into one or more of the categories: wrong, weird, unrealistic, or other.  
    Use the following definitions:  
    - **Wrong**: The image contains mistakes or errors, such as incorrect object placements, impossible physics, or factual inaccuracies.  
    - **Weird**: The image is very strange, unusual, unexpected, or not natural. It might include surreal combinations of elements, strange colors or shapes, or abstract and bizarre arrangements.  
    - **Unrealistic**: The image is not based on facts or real-world logic. It could feature fantastical or imaginary elements, implausible lighting or shadows, or overly exaggerated details.  
    - **Other**: Use this label only if the image does not fall into any of the other three categories (Wrong, Weird, or Unrealistic).  

    **Before classifying, you will first generate three refined and natural-sounding questions** to help analyze the image. These questions should be based on the following original ideas:

    1. Does the image contain mistakes or impossible elements?  
    2. Is the image strange, unusual, or surreal in any way?  
    3. Does the image look imaginary or unrealistic?

    You should rewrite the above reference questions using fluent, natural English phrasing that retains their original meaning. Then, answer those questions based on the image and provide your final classification.

    Format your response as follows:

    **Generated Questions:**  
    Q1: <your rephrased question 1>  
    Q2: <your rephrased question 2>  
    Q3: <your rephrased question 3>

    **Answers:**  
    A1: <answer to Q1>  
    A2: <answer to Q2>  
    A3: <answer to Q3>

    **Categories:** <comma-separated list of applicable categories>
    """


    def get_model_response(prompt_text):
        chat_completion = client.chat.completions.create(
            model="llama-3.2-90b-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_image}"
                            }
                        },
                    ],
                }
            ],
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stream=False,
            stop=None,
        )
        return chat_completion.choices[0].message.content.strip()

    # **First attempt**
    response_text = get_model_response(prompt_template)

    # Extract categories from response
    qna_parts = response_text.split("Categories:")
    categories_part = qna_parts[1].strip() if len(qna_parts) > 1 else ""

    # Validate categories
    valid_labels = {"wrong", "weird", "unrealistic", "other"}
    cleaned_labels = [
        word.strip().lower().rstrip(".")
        for word in categories_part.split(",")
        if word.strip().lower().rstrip(".") in valid_labels
    ]

    # **If empty, retry with a revised prompt**
    if not cleaned_labels:
        print(f"[DEBUG] Empty prediction for {filename}. Retrying...")
        retry_prompt = f"""
        The model did not classify the image into any category in the first attempt.
        Re-evaluate the image carefully and classify it into at least one of the following: 
        Wrong, Weird, Unrealistic, or Other.
        If the image is difficult to classify, choose the closest match.

        Original response: {response_text}

        **Generated Questions & Answers:**  
        """
        response_text = get_model_response(retry_prompt)

        # Extract new classification
        qna_parts = response_text.split("Categories:")
        categories_part = qna_parts[1].strip() if len(qna_parts) > 1 else ""

        cleaned_labels = [
            word.strip().lower().rstrip(".")
            for word in categories_part.split(",")
            if word.strip().lower().rstrip(".") in valid_labels
        ]

    # **Logging model responses**
    analysis_filename = "model_analysis.txt"
    with open(analysis_filename, "a", encoding="utf-8") as file:
        file.write(f"Filename: {filename}\n")
        file.write(f"{response_text}\n")
        file.write(f"Ground Truth: {ground_truth_labels}\n")
        file.write(f"Model Prediction: {cleaned_labels}\n")
        file.write("-" * 50 + "\n")

    # **Print debugging information**
    print(f"Model Analysis for {filename}:\n{response_text}\n")
    print(f"Ground Truth: {ground_truth_labels}")
    print(f"Model Prediction: {cleaned_labels}\n" + "-" * 50)

    return cleaned_labels


# Load labeled data
with open('images_labels.json', 'r') as f:
    labeled_data = json.load(f)

# Collect results for comparison
results = []

# Image folder path
image_folder = os.environ.get("IMAGE_DIR", "images")

for entry in labeled_data:
    image_path = f"{image_folder}\\{entry['filename']}"  # Build the full path to the image
    ground_truth_labels = entry["type"] if isinstance(entry["type"], list) else [entry["type"]]

    # Normalize ground truth labels to lowercase
    ground_truth_labels = [label.lower() for label in ground_truth_labels]

    # Encode image to base64
    encoded_image = encode_image(image_path)

    # Get the model's prediction using Groq Vision API with base64 image
    model_labels = classify_image_with_groq_base64(encoded_image, entry["filename"], ground_truth_labels)

    # Print predictions, filename, and ground truth for debugging
    print(f"Filename: {entry['filename']}")
    print(f"  Ground Truth: {ground_truth_labels}")
    print(f"  Model Prediction: {model_labels}")
    print("-" * 50)

    # Store results
    results.append({
        "id": entry["id"],
        "filename": entry["filename"],
        "ground_truth": ground_truth_labels,
        "model_prediction": model_labels
    })

# Evaluate performance
true_labels = [set(r["ground_truth"]) for r in results]  # Convert ground truth to sets
predicted_labels = [set(r["model_prediction"]) for r in results]  # Convert predictions to sets

# Calculate exact match accuracy
exact_match_count = sum(1 for t, p in zip(true_labels, predicted_labels) if t == p)
accuracy = exact_match_count / len(true_labels)

all_types = ["wrong", "weird", "unrealistic", "other"]
# Flatten labels for confusion matrix
flat_true_labels = []
flat_predicted_labels = []

for true_set, pred_set in zip(true_labels, predicted_labels):
    for label in all_types:
        flat_true_labels.append(label in true_set)
        flat_predicted_labels.append(label in pred_set)

# Generate confusion matrix
conf_matrix = confusion_matrix(flat_true_labels, flat_predicted_labels, labels=[True, False])

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["True", "False"], yticklabels=["True", "False"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

# Calculate additional metrics
type_metrics = []
for label_type in all_types:
    y_true = [label_type in t for t in true_labels]
    y_pred = [label_type in p for p in predicted_labels]
    precision = precision_score(y_true, y_pred, zero_division=1)
    recall = recall_score(y_true, y_pred, zero_division=1)
    f1 = f1_score(y_true, y_pred, zero_division=1)
    type_metrics.append({
        "Type": label_type,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

# Create a DataFrame for metrics and save to Excel
metrics_df = pd.DataFrame(type_metrics)
metrics_df.to_excel("classification_metrics.xlsx", index=False)

# Print overall accuracy and metrics
print("Overall Exact Match Accuracy:", accuracy)
print("\nPer-Type Metrics:")
print(metrics_df)
