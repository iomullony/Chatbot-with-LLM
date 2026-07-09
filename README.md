Sánchez-O'Mullony Martínez, María Isabel, 123456

This project implements a local Retrieval-Augmented Generation chatbot for the THD/DIT International Office using Streamlit, sentence-transformers, and flan-t5-small.

# Chatbot long description

## Task

The goal of this project was to build a Streamlit-based chatbot capable of answering questions about the Deggendorf Institute of Technology (DIT/THD), with a focus on the International Office. The target users are prospective and current international students asking about exchange programmes, application processes, language requirements, orientation, and contact information.

## Approach

The chatbot uses Retrieval-Augmented Generation (RAG) with a local Hugging Face language model, google/flan-t5-small. The knowledge base is stored in [`knowledge_base.txt`](knowledge_base.txt) and is split into short paragraphs so the retriever can select the most relevant facts for each user question.

The pipeline works as follows:

1. **Knowledge base**: A manually curated [`knowledge_base.txt`](knowledge_base.txt) file containing factual information about the DIT International Office, sourced from the official THD website. The file is split into small paragraphs so the retriever can select the most relevant facts for each user question.

2. **Retrieval**: When the user asks a question, the `sentence-transformers` model (`all-MiniLM-L6-v2`) encodes both the question and all knowledge base chunks into vector embeddings. Cosine similarity is computed between the question embedding and all chunk embeddings to retrieve the top 3 most relevant chunks.

3. **Generation**: The retrieved chunks are injected into a structured prompt together with the user's question. The prompt instructs `flan-t5-small` to answer only based on the provided context. The model then generates a response using `AutoModelForSeq2SeqLM`.

4. **Logging**: Every user message and assistant response is automatically appended to `chatbot_history.log` using Python's `logging` module, with timestamps and session separators.

5. **UI**: The Streamlit interface follows a standard chat layout with message history. A sidebar toggle allows the user to inspect which knowledge base chunks were retrieved for each answer, which is useful for debugging.

The speech recognition and text-to-speech components from the original script were removed because microphone access was not available in the development environment.

# Run

Make sure all dependencies are installed:

```
pip install -r requirements.txt
```

Then start the Streamlit app:

```
streamlit run app.py
```

On first run, the models (`all-MiniLM-L6-v2` and `google/flan-t5-small`) will be downloaded automatically from Hugging Face (~400 MB total) and cached locally. Subsequent starts are faster.

# Hardware and runtime environment

- **CPU**: Intel(R) Xeon(R) Silver 4214 @ 2.20 GHz, 4 virtual cores (1 thread per core)
- **RAM**: 15 GB total, ~12 GB available during development
- **GPU**: None — all inference runs on CPU

Running entirely on CPU means model loading takes approximately 30–60 seconds on first start. Response generation typically takes 5–15 seconds per query depending on prompt length. Despite the lack of GPU, the small model sizes of `flan-t5-small` and `all-MiniLM-L6-v2` make the system usable in practice.

# Discussion of the results

The chatbot performs well for direct factual questions such as application deadlines, contact persons, and language requirements. It also benefits from the retrieval step because it can ground its answers in the provided knowledge base instead of relying purely on the language model's general knowledge.

At the same time, the model still has limits. It can struggle with follow-up questions that require context across turns, and it may sometimes select the wrong email address or phone number when multiple similar facts appear in the retrieved chunks. These issues are typical of a small local model such as flan-t5-small.

The following example dialogs are taken from `example-dialogs.txt` and reflect the current version of the chatbot. They are ordered in the same way as the example file and illustrate both the strengths of the system and the areas where it still remains limited.

---
DIALOG 1: Incoming Exchange Student — Application Deadlines & Process
Target: International Office (Incomings & Deadlines)
---

This dialogue shows that the chatbot can now answer the application timeline and the online-document process clearly and consistently.

```
USER:  Hello! I want to come to DIT as an exchange student for the winter semester. When is the application period?
ASSISTANT: 1 April to 1 June for October (winter semester) entries, and 1 October to 1 December for March (summer semester) entries
USER: Can I just go ahead and fill out the application form on your website right now?
ASSISTANT: The application process is done entirely online via a system called Mobility Online.
USER: Do I need to mail my physical documents to Deggendorf after completing the form?
ASSISTANT: No
```

**Analysis**: This is a fairly strong example because the chatbot correctly answers the key question about the application period and clearly explains that the application process is handled online. The final answer is also relevant, and the dialogue shows that the system can provide useful procedural information when the user asks direct questions about the application workflow.

---

DIALOG 2: Exchange Programmes & Language Requirements
Target: International Office (Programmes & Requirements)
---

This example shows that the chatbot can provide useful information about available programmes and the language expectations for exchange students.

```
USER:  Hi, I'm a Computer Science student. Are there any specific English-taught programmes available for exchange students on the Deggendorf campus?
ASSISTANT: yes
USER: What are the language proficiency requirements to qualify for this exchange?
ASSISTANT: English or German
USER: Do I need to submit an official language certificate like TOEFL or IELTS for admission?
ASSISTANT: To qualify for an exchange programme at DIT, students must have completed at least two semesters at their home university and have a good to excellent knowledge of the language of instruction (English or German), at least level B1.
```

**Analysis**: This dialogue shows that the chatbot can identify the topic of exchange programmes and language requirements, but the answers are not always precise. The first response is broad and informative, while the second and third turns show that the model sometimes mixes related ideas together instead of answering the specific user question directly. This is a good example of the system being generally helpful but still somewhat imprecise.

---

DIALOG 3: Language Centre & TestDaF Fees
Target: International Office / Language Centre
---

This conversation demonstrates that the chatbot can answer subject-specific questions about German courses and official exams fairly well.

```
USER:  Hello! Are exchange students required to learn German while studying at DIT?
ASSISTANT: To qualify for an exchange programme at DIT, students must have completed at least two semesters at their home university and have a good to excellent knowledge of the language of instruction (English or German), at least level B1.
USER: Can I take an official language exam like the TestDaF at the university?
ASSISTANT: Not required
USER: Whom should I contact if I want to register for the exam?
ASSISTANT: Students can obtain official German language certificates at DIT, including TestDaF and telc exams.
```

**Analysis**: This example is weaker than the earlier ones because the chatbot does not fully answer the user's intent. The first answer drifts away from the original question about German-language requirements, the second answer is not directly related to the TestDaF question, and the third response does not provide a concrete contact point. It shows that the model can retrieve related background information, but it still struggles to stay focused on the specific task.

---

DIALOG 4: Orientation Week & Arrival Support
Target: International Office (Arrival & Orientation)
---

This dialogue shows how the chatbot can respond well to practical questions about arrival and orientation activities.

```
USER:  Hi, I am planning my arrival for the upcoming semester. Is the Orientation Week mandatory?
ASSISTANT: Orientation Week is a compulsory event for all exchange students at the main campus in Deggendorf.
USER: What kind of activities are organized during this week?
ASSISTANT: Leisure activities include horseback riding, canoeing on the Danube, hiking in the Bavarian Forest, and skiing in winter
USER: Who can I email if I have specific questions about incoming exchange logistics?
ASSISTANT: the International Office.
```

**Analysis**: This dialogue illustrates a clear weakness of the system. The first answer about Orientation Week is relevant, but the second answer drifts to unrelated leisure activities instead of describing the actual programme activities. The final answer is also too vague and does not provide a concrete contact detail. This shows that the chatbot can identify the topic, but it does not always stay focused on the user’s real intent.

---

DIALOG 5: Specific Contact Persons for Programmes
Target: International Office (Staff & Contacts)
---

This final dialogue highlights the chatbot's ability to provide programme-specific contact information.

```
USER:  Hello, I have a detailed question about the General Engineering exchange programme structure. Who is the coordinator for this?
ASSISTANT: incomings@th-deg.de
USER: What is her telephone number?
ASSISTANT: +49 991 3615-0
USER: What is her telephone number of Nadine Kappl if I need to call her?
ASSISTANT: +49 991 3615-0
USER: And who is the contact person for the General Business programme?
ASSISTANT: Laura Schümann
```

**Analysis**: This is one of the weaker dialogues because the chatbot does not answer the first question correctly and instead returns an email address that is unrelated to the requested coordinator. The later answers also remain too generic and do not provide the specific contact information the user is looking for. This shows that the model still has difficulty with exact staff and contact details, especially when several similar facts are present in the knowledge base.

# Description of technical problems during the development process

- **`text2text-generation` pipeline error**: The installed version of `transformers` did not support the `text2text-generation` task string in the `pipeline()` constructor. This was resolved by loading `flan-t5-small` directly using `AutoTokenizer` and `AutoModelForSeq2SeqLM` and calling `model.generate()` manually.

- **Streamlit warnings**: On startup, Streamlit's file watcher tried to inspect all submodules of the `transformers` library, including vision models that require `torchvision`. Since `torchvision` was not installed, this produced a stream of `ModuleNotFoundError` warnings, but these did not affect functionality.

- **Microphone unavailable in VM**: The speech recognition and TTS features from the original `code_02_chatbot.py` could not be tested or used, as the virtual machine environment does not expose a microphone device. These components were removed from the final implementation.

# Suggestions for improvements

- **Larger or more capable model**: `flan-t5-small` is very limited in its ability to reason across multiple turns and disambiguate between similar facts (e.g. multiple phone numbers or email addresses). A larger model such as `flan-t5-base` or `SmolLM-135M-Instruct` would likely produce more accurate and context-aware answers.

- **Conversation history in the prompt**: Currently, each prompt only contains the retrieved chunks and the current question. Including the last 2–3 turns of conversation history in the prompt would allow the model to handle follow-up questions and contextual references (e.g. "but it is july") much more reliably.

- **Larger and more structured knowledge base**: The knowledge base could be expanded with more specific information, such as individual course descriptions, partner university lists, and FAQs sourced directly from the official THD website. Shorter, more focused chunks would also improve retrieval precision.

- **Re-ranking retrieved chunks**: Instead of using only cosine similarity for retrieval, a cross-encoder re-ranker could be added to re-score the top-k candidates and select the most relevant chunk more accurately — reducing cases where the wrong phone number or email is retrieved.

- **Adding speech input/output**: Once a microphone is available, the original voice recognition and TTS components from `code_02_chatbot.py` could be re-integrated, making the chatbot more accessible and interactive.