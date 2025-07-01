from sklearn.metrics.pairwise import cosine_similarity
import spacy
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModel, pipeline
from sentence_transformers import SentenceTransformer, util


nlp = spacy.load("en_core_web_md")


# "Nucha/Nucha_ITSkillNER_BERT"
model_name = "algiraldohe/lm-ner-linkedin-skills-recognition"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

ner = pipeline("ner", model=model, tokenizer=tokenizer,
               aggregation_strategy="simple")


# sentence-transformers/all-roberta-large-v1
embed_model = SentenceTransformer("all-mpnet-base-v2")

cs_cv = """

Dr. Evelyn Carter

Professor of Computer Science
notre dame university -loiauze

Contact Information
Email: ecarter@ndu.edu.lb

Office: Turing Hall, Room 405

Phone: (555) 123-4567

Education
Ph.D. in Computer Science, Stanford University (2010)

Dissertation: "Quantum-Inspired Algorithms for Large-Scale Optimization"

M.Sc. in Artificial Intelligence, MIT (2006)

B.Sc. in Computer Science, University of Cambridge (2004)

Academic Appointments
Full Professor, notre dame university-louaize (2020-Present)

Associate Professor, notre dame university-louaize (2015-2020)

Assistant Professor, notre dame university-louaize (2010-2015)

Research Interests
Quantum Computing

Machine Learning & Neural Networks

Algorithm Design & Optimization

Cybersecurity & Cryptography

Selected Publications
Carter, E., & Zhang, L. (2023). "Hybrid Quantum-Classical Neural Networks for Secure Data Classification." Nature Computing Science.

Carter, E., et al. (2021). "Optimizing Deep Learning Models for Edge Devices." Journal of AI Research.

Carter, E. (2018). "A Novel Approach to Post-Quantum Cryptography." ACM Transactions on Security.

Teaching Experience
Advanced Machine Learning (Graduate Level)

Introduction to Quantum Computing (Undergraduate/Graduate)

Algorithms & Data Structures (Undergraduate)

Cybersecurity Fundamentals (Undergraduate)

Awards & Honors
ACM Distinguished Scientist (2022)

notre dame university-louaize Excellence in Teaching Award (2019)

NSF CAREER Award (2016)

Professional Service
Program Chair, International Conference on Quantum Computing (ICQC 2024)

Reviewer for IEEE Transactions on Neural Networks

"""

cs_cv_structured = """
Dr. Evelyn Carter, Professor of Computer Science
Department of Computer Science, Notre Dame University - Louaize

##Contact Information
Email: ecarter@ndu.edu.lb
Office: Turing Hall, Room 405
Phone: (555) 123-4567

##Education
Ph.D. in Computer Science, Stanford University (2010)
M.Sc. in Artificial Intelligence, MIT (2006)
B.Sc. in Computer Science, University of Cambridge (2004)

##Teaching Experience
Notre Dame University - Louaize
CSC 600: Advanced Machine Learning (Graduate)
CSC 450: Introduction to Quantum Computing (Undergraduate/Graduate)
CSC 210: Algorithms & Data Structures (Undergraduate)
CSC 320: Cybersecurity Fundamentals (Undergraduate)

##Publications (Selected)
Carter, E., & Zhang, L. (2023). "Hybrid Quantum-Classical Neural Networks for Secure Data Classification." Nature Computing Science.
Carter, E., et al. (2021). "Optimizing Deep Learning Models for Edge Devices." Journal of AI Research.
Carter, E. (2018). "A Novel Approach to Post-Quantum Cryptography." ACM Transactions on Security.

##Skills & Expertise
Quantum Computing: Quantum algorithms, Quantum machine learning
Machine Learning: Deep learning, Neural networks, Model optimization
Cybersecurity: Cryptography, Post-quantum cryptography
Algorithms: Design, Optimization, Data structures

"""


mathematics_cv = """

Dr. Mahmoud Atallah
Professor of Mathematics
Al-Farabi University

Email: matallah@alfarabi.edu.lb
Office: Math Building, Room 207
Phone: (555) 665-8832

Education
Ph.D. in Pure Mathematics (Topology), University of Cambridge (2008)
M.Sc. in Applied Mathematics, American University of Cairo (2003)

Teaching Experience
- Real Analysis
- Abstract Algebra
- Topology and Geometry
- Mathematical Logic

Research Interests
- Algebraic Topology
- Number Theory
- Mathematical Modeling in Epidemiology

Publications
Atallah, M. (2022). “Fixed-Point Theorems in Generalized Topological Spaces.” Journal of Pure Math.
Atallah, M. (2019). “Combinatorial Methods in Number Theory.”

Academic Activities
Organizer - Annual Symposium on Pure Math
Editor - Middle East Mathematics Review

"""


english_cv = """

Dr. Jonathan P. Whitmore
Professor of English Literature
Department of English & Comparative Literature
Kingsbridge University (fictional)
Email: j.whitmore@kingsbridge.edu | Phone: (555) 123-4567

Education
Ph.D. in English Literature - Oxford University, 2005
Dissertation: "The Haunted Text: Spectrality and Memory in Postmodern British Fiction"

M.A. in English Literature - University of Cambridge, 2001

B.A. (Hons) in English - King's College London, 1999

Academic Appointments
Full Professor of English Literature (2018-Present)
Kingsbridge University, UK

Associate Professor (2012-2018)
Kingsbridge University, UK

Assistant Professor (2006-2012)
University of Northshire (fictional), UK

Visiting Lecturer (2004-2006)
Oxford University, UK

Research Interests
Postmodern & Contemporary British Fiction

Gothic Literature & Spectrality in Narrative

Memory, Trauma, and Textual Haunting

Digital Humanities & Literary Analysis

Selected Publications
Books
Whitmore, J. (2017). Phantoms of the Page: Ghosts, Memory, and the Novel. Cambridge University Press.

Whitmore, J. (2011). The Unwritten Past: Postmodern Hauntings in British Fiction. Palgrave Macmillan.

Peer-Reviewed Articles
"Digital Echoes: Algorithmic Hauntings in Hypertext Literature" (2023). Modern Fiction Studies, 69(2), 45-67.

"The Ghost in the Machine: AI and the Future of Literary Analysis" (2021). New Literary History, 52(3), 321-340.

"Revenants of the Real: Spectral Memory in Ian McEwan's Atonement" (2018). Contemporary Literature, 59(4), 512-535.

Edited Collections
Co-editor, The Palgrave Handbook of Gothic Literature (2020). Palgrave Macmillan.

Teaching Experience
Undergraduate Courses
ENG 301: Postmodern British Fiction

ENG 245: Gothic Literature: From Walpole to Winterson

ENG 150: Critical Theory & Literary Analysis

Graduate Seminars
ENG 701: Memory & Narrative in Contemporary Fiction

ENG 720: Digital Humanities & Literary Studies

Awards & Honors
Kingsbridge Distinguished Scholar Award (2022)

British Academy Research Fellowship (2016-2017)

Modern Language Association (MLA) Prize for Best First Book (2012)

Professional Service
Editorial Board Member, Journal of Contemporary Gothic Studies (2019-Present)

Peer Reviewer for Modern Fiction Studies, Textual Practice, Gothic Studies

Chair, Kingsbridge University Faculty Senate (2020-2022)

Grants & Fellowships
British Research Council Grant (2020-2023) - "AI and the Future of Narrative" (£150,000)

Kingsbridge Digital Humanities Initiative (2018) - "Algorithmic Close Reading" (£50,000)

Conference Presentations
"Haunted Algorithms: AI as Literary Specter" - MLA Annual Convention, 2024

"The Digital Uncanny" - International Gothic Association Conference, 2022

"Postmemory & the Novel" - British Association for Contemporary Literary Studies, 2019

Languages
English (Native)

French (Advanced)

Latin (Reading Proficiency)

"""

course_cs = """

Data Structures - CSC 313
This course is a detailed coverage of standard data structures with an emphasis on complexity analysis. Topics include: Asymptotic analysis, linked lists, stacks, queues, trees and balanced trees, hashing, priority queues and heaps, sorting. Standard graph algorithms such as DFS, BFS, shortest paths and minimum spanning trees are also covered. Prerequisites: CSC 213 or CSC 215.

"""

course_math = """

Discrete Mathematics - MAT 211
This course describes arithmetic in different bases, set theory, relations and functions, mathematical reasoning and induction, counting techniques, permutations and combinations, logic, Boolean algebra; and lattice theory.
Prerequisite: Sophomore Standing.

"""

ruler = nlp.add_pipe("entity_ruler", before="ner")

patterns = [
    {
        "label": "EMAIL",
        "pattern": [
            {"TEXT": {
                "REGEX": r"^[\w.-]+@[a-zA-Z]+\.[a-zA-Z]{2,}(?:[\.][a-zA-Z]{2,})?$"
            }
            }
        ]
    },
    {
        "label": "COURSE_ID",
        "pattern": [

            {"TEXT": {"REGEX": r"^[A-Za-z]{2,4}$"}},
            {"IS_SPACE": True, "OP": "?"},
            {"TEXT": {"REGEX": r"^\d{2,4}$"}}
        ]
    }
]

ruler.add_patterns(patterns)


def func_cv(cv):

    doc_professor = nlp(cv)

    for ent in doc_professor.ents:
        if ent.label_ == "PERSON":
            name_professor = ent.text
            break

    for ent in doc_professor.ents:
        if ent.label_ == "EMAIL":
            email_professor = ent.text
            break

    print(f"professor name: {name_professor}")

    print(f"professor email: {email_professor}")

    entities = ner(cv)
    skills = {ent['word'] for ent in entities if ent['entity_group']
              in {"TECHNICAL", "TECHNOLOGY", "BUSINESS", "SOFT"}}
    print("Extracted Skills:")
    for skill in skills:
        print("-", skill)


def skip_line(line: str):
    """Return True if line is metadata or irrelevant."""
    line = line.lower()
    return any(kw in line for kw in [
        "email", "phone", "office", "contact", "address", "reviewer", "room", "turing hall"
    ]) or len(line.strip()) < 5


def extract_relevant_cv_sentences(cv_text, course_text, top_n=5):

    lines = [
        line.strip()
        for line in cv_text.split("\n")
        if len(line.strip()) > 5 and not skip_line(line)
    ]

    course_embedding = embed_model.encode(course_text, convert_to_tensor=True)

    scored_lines = []
    for line in lines:
        line_embedding = embed_model.encode(line, convert_to_tensor=True)
        sim = util.cos_sim(line_embedding, course_embedding).item()
        scored_lines.append((sim, line))

    top_lines = sorted(scored_lines, key=lambda x: x[0], reverse=True)[:top_n]

    for score, sentence in top_lines:
        print(f"Score: {score:.3f} | Sentence: {sentence}")

    return [line for _, line in top_lines]


def unstructured_cv_course_similarity(cv_text, course_text):  # unstructured CV
    top_sentences = extract_relevant_cv_sentences(
        cv_text, course_text, top_n=5)

    combined_text = " ".join(top_sentences)
    combined_embedding = embed_model.encode(
        combined_text, convert_to_tensor=True)
    course_embedding = embed_model.encode(course_text, convert_to_tensor=True)

    similarity_score = util.cos_sim(
        combined_embedding, course_embedding).item()
    return round(similarity_score, 3)


"""****************************************************************************************************************************************"""


def extract_section(cv_text, section_name):

    try:

        section_part = cv_text.split(f"##{section_name}")[1]
        section_content = section_part.split("##")[0].strip()
        return section_content
    except IndexError:
        return ""


def extract_publication_titles(cv_text):

    publications = extract_section(cv_text, "Publications (Selected)")
    if not publications:
        return ""

    titles = re.findall(r'"(.*?)"', publications)
    return " ".join(titles)


def structured_cv_course_similarity(cv_text, course_text):

    teaching = extract_section(cv_text, "Teaching Experience")
    skills = extract_section(cv_text, "Skills & Expertise")
    publications = extract_publication_titles(cv_text)

    combined = f"""
    TEACHING: {teaching}
    SKILLS: {skills}
    PUBLICATIONS: {publications}
    """

    print(combined)

    combined_embedding = embed_model.encode(combined)
    course_embedding = embed_model.encode(course_text)

    similarity_score = util.cos_sim(
        combined_embedding, course_embedding).item()
    return round(similarity_score, 3)


"""****************************************************************************************************************************************"""

# print("CV: \n")
# func_cv(abdo_abou_jaoude_cv)
# print("Course Description: \n")
# func_course(course_math)
print(
    f"similarity score: {unstructured_cv_course_similarity(english_cv, course_cs)}")
print(
    f"similarity score: {structured_cv_course_similarity(cs_cv_structured, course_cs)}")
