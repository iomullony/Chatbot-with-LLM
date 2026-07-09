Sánchez-O'Mullony Martínez, María Isabel, 123456

THD/DIT International Office Chatbot — RAG-based Q&A assistant using flan-t5-small

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

After the latest updates in the app and the shorter knowledge-base chunks, the newest test runs in `chatbot_history.log` show a slight improvement for direct factual questions. The chatbot answered some questions about Orientation Week and contact information more directly, and it also continued to provide correct answers for application deadlines and programme contacts. However, some vague or follow-up prompts still produced incomplete or off-topic replies, which shows that the model still needs a stronger memory mechanism or a larger model for more reliable multi-turn reasoning.

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
ASSISTANT: Yes
```

**Analysis**: This dialogue shows that the chatbot can retrieve the main facts about the application period and the online application process. The first answers are generally aligned with the knowledge base, but the final response becomes less reliable and suggests that the model is not fully consistent when handling a follow-up question. This is a good example of how the system can answer procedural questions well at first, while still struggling with precision in later turns.

---

DIALOG 2: Exchange Programmes & Language Requirements
Target: International Office (Programmes & Requirements)
---

This example shows that the chatbot can provide useful information about available programmes and the language expectations for exchange students.

```
USER:  Hi, I'm a Computer Science student. Are there any specific English-taught programmes available for exchange students on the Deggendorf campus?
ASSISTANT: DIT offers several English-taught exchange programmes for incoming students. On the Deggendorf campus these include: General Business (for business Bachelor students), General Engineering (for engineering Bachelor students), International Computer Science (for computer science Bachelor students, not available for free movers due to limited capacity), and General Health (for healthcare sciences Bachelor students). All these programmes are 100% in English and are free of charge for Bachelor students. Students can earn up to 30 ECTS credits per semester. The winter semester starts on 1 October and the summer semester starts on 15 March. - To qualify for an exchange programme at DIT, students must have completed at least two semesters at their home university and have a good to excellent knowledge of the language of instruction (English or German), at least level B1.
USER: What are the language proficiency requirements to qualify for this exchange?
ASSISTANT: All exchange students are required to participate in German language courses during their semester.
USER: Do I need to submit an official language certificate like TOEFL or IELTS for admission?
ASSISTANT: To qualify for an exchange programme at DIT, students must have completed at least two semesters at their home university and have a good to excellent knowledge of the language of instruction (English or German), at least level B1.
```

**Analysis**: This dialogue shows that the chatbot can retrieve a broad set of relevant facts about exchange programmes and language requirements. It correctly mentions the existence of English-taught options and the B1-level expectation, but it also mixes several related points into one answer and does not always stay precise. This is a useful example of the system being informative while still being somewhat broad and less focused than ideal.

---

DIALOG 3: Language Centre & TestDaF Fees
Target: International Office / Language Centre
---

This conversation demonstrates that the chatbot can answer subject-specific questions about German courses and official exams fairly well.

```
USER:  Hello! Are exchange students required to learn German while studying at DIT?
ASSISTANT: All exchange students are required to participate in German language courses during their semester.
USER: Can I take an official language exam like the TestDaF at the university?
ASSISTANT: Students can obtain official German language certificates at DIT, including TestDaF and telc exams.
USER: Whom should I contact if I want to register for the exam?
ASSISTANT: Students can obtain official German language certificates at DIT, including TestDaF and telc exams.
```

**Analysis**: This example is partly successful because the chatbot identifies the general German-language requirement and the presence of official exams such as TestDaF. However, it does not answer the final question as specifically as it could, since it repeats general information rather than giving a concrete contact point. It shows that the model can retrieve relevant background information, but still struggles to turn it into a precise, user-specific answer.

---

DIALOG 4: Orientation Week & Arrival Support
Target: International Office (Arrival & Orientation)
---

This dialogue shows how the chatbot can respond well to practical questions about arrival and orientation activities.

```
USER:  Hi, I am planning my arrival for the upcoming semester. Is the Orientation Week mandatory?
ASSISTANT: The Orientation Week takes place one to two weeks before the start of every semester, annually in March and September. Activities in The Orientation Week include a guided campus and town tour, IT and library introduction, registering at the residents' registration office, obtaining a residence permit, intercultural training, a Bavarian Evening, and various day trips.
USER: What kind of activities are organized during this week?
ASSISTANT: Leisure activities include horseback riding, canoeing on the Danube, hiking in the Bavarian Forest, and skiing in winter.
USER: Who can I email if I have specific questions about incoming exchange logistics?
ASSISTANT: The correct email address to contact the International Office for incoming exchange students is incomings@th-deg.de.
```

**Analysis**: The chatbot handles the first question reasonably well and gives a detailed explanation of Orientation Week. In the second turn, however, it drifts to unrelated leisure activities, which shows that the retrieval and generation process can lose focus when the user asks a follow-up question. The final answer about the contact email is useful, but the overall dialogue shows that topical relevance is still inconsistent.

---

DIALOG 5: Specific Contact Persons for Programmes
Target: International Office (Staff & Contacts)
---

This final dialogue highlights the chatbot's ability to provide programme-specific contact information.

```
USER:  Hello, I have a detailed question about the General Engineering exchange programme structure. Who is the coordinator for this?
ASSISTANT: Johanna Maurer
USER: What is her telephone number?
ASSISTANT: +49 991 3615-0
USER: What is her telephone number of Nadine Kappl if I need to call her?
ASSISTANT: +49 991 3615-0
USER: And who is the contact person for the General Business programme?
ASSISTANT: Laura Schümann
```

**Analysis**: This dialogue shows that the chatbot can retrieve programme-related staff information, but it does not always answer with the most relevant detail. The assistant gives a name and a phone number, yet the response remains somewhat generic and does not fully align with the specific request. This highlights that the model can access the right topic but still has difficulty with precision and consistency in more detailed contact questions.

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