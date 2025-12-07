🚀 QUICK START GUIDE
Install dependencies:

bash
pip install -r requirements.txt
Basic usage:

python
from turnilab import TurniLab

# Initialize
tl = TurniLab()

# Analyze text
result = tl.analyze("Your text here")
print(f"AI likelihood: {result['ai_score']}")

# Transform text (humanize)
transformed = tl.transform.run("Your AI text here")
print(f"Humanized: {transformed}")

# Train on your dataset
human_texts = [...]  # List of human-written texts
ai_texts = [...]     # List of AI-generated texts
metrics = tl.train_on_dataset(human_texts, ai_texts, save_path="my_model.pkl")
Command line usage:

bash
# Demo
python turnilab.py --mode demo

# Analyze text
python turnilab.py --mode analyze --text "Your text here"

# Train model
python turnilab.py --mode train --human-file human.txt --ai-file ai.txt

# Transform text
python turnilab.py --mode transform --text "AI text to humanize" --output transformed.json
📊 TRAINING WITH YOUR DATA
Prepare two text files:

human.txt - Human-written texts (one per line)

ai.txt - AI-generated texts (one per line)

Train the model:

python
tl = TurniLab()

with open("human.txt", "r") as f:
    human_texts = [line.strip() for line in f if line.strip()]

with open("ai.txt", "r") as f:
    ai_texts = [line.strip() for line in f if line.strip()]

metrics = tl.train_on_dataset(human_texts, ai_texts)
print(f"Model trained with AUC: {metrics['test_auc']:.3f}")
This complete system provides:

✅ Full linguistic analysis

✅ Perplexity and burstiness calculation

✅ Text transformation/humanization

✅ Trainable local classifier

✅ Model persistence

✅ Command line interface

✅ Comprehensive feature extraction

The system is designed for academic research purposes to understand AI text detection mechanisms and develop humanization techniques.