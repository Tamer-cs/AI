import spacy
from spacy.matcher import Matcher
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline


nlp = spacy.load("en_core_web_md")


# "Nucha/Nucha_ITSkillNER_BERT"
model_name = "algiraldohe/lm-ner-linkedin-skills-recognition"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

ner = pipeline("ner", model=model, tokenizer=tokenizer,
               aggregation_strategy="simple")

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

history_cv = """

Mr. Jonathan Meyers

History Teacher
Greenwood High School

Contact Information
Email: jmeyers@greenwood.edu.lb
Office: Humanities Building, Room 212
Phone: (555) 987-6543

Education
M.A. in History, University of Chicago (2012)
B.A. in History & Political Science, University of Michigan (2009)

Teaching Experience
World History (High School)
European History (Advanced Placement)
American History and Government
Historical Research Methods

Research Interests
Ancient Civilizations
World War II Strategy & Politics
Middle Eastern History
Colonialism and Post-Colonial Studies

Publications & Projects
Meyers, J. (2021). "Teaching WWII Through Multimedia Narratives." Journal of Historical Education.
Meyers, J. (2019). "Decolonizing the Curriculum: A High School Approach." Presented at the National History Teachers Conference.

Awards
Greenwood Teacher of the Year (2020)
Illinois Excellence in Social Studies Education (2017)

Professional Involvement
Member, National Council for History Education (NCHE)
Curriculum Advisor, Greenwood District Education Board

"""

biology_cv = """

Dr. Nour El-Haddad
Assistant Professor - Molecular Biology
Faculty of Science, Phoenicia University

Email: nour.haddad@phoenicia.edu.lb
Office: Science Complex, Lab 6A
Phone: (555) 778-4455

Education
Ph.D. in Molecular Genetics, Stanford University (2017)
B.Sc. in Biology, Lebanese University (2011)

Teaching Experience
- Molecular Biology
- Genetics and Cell Biology
- Research Methods in Biology
- Genetic Engineering

Research Interests
- CRISPR-Cas9 Applications
- Epigenetics and Cancer Therapy
- Genomic Data Analysis

Publications
El-Haddad, N. (2023). “CRISPR Editing Efficiency in Mammalian Cells.” Nature Biotechnology.
El-Haddad, N. (2021). “Non-coding RNA in Tumor Progression.” Cancer Research Letters.

Grants & Awards
NIH Research Grant (2022)
Best Poster - EMBL Genomics Conference (2020)

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

arts_cv = """

Prof. Layla Zahran
Professor of Visual Arts
Beirut College of Arts and Design

Email: l.zahran@bcad.edu.lb
Office: Visual Arts Building, Room 2F
Phone: (555) 901-4478

Education
M.F.A. in Visual Communication, School of the Art Institute of Chicago (2012)
B.F.A. in Graphic Design, LAU (2007)

Teaching Experience
- Visual Storytelling
- Graphic Design Studio
- History of Modern Art
- Typography and Communication

Research Interests
- Post-war Arab Visual Culture
- Digital Art & Identity
- Feminist Aesthetics

Exhibitions & Projects
Zahran, L. (2023). “Beirut Fragments” - Solo Exhibition, Sursock Museum
Zahran, L. (2020). “Designing Resistance” - Collaborative Project, AUB Art Gallery

Awards
Arab Women in Arts Prize (2022)
National Design Award - Typography Category (2021)

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

abdo_abou_jaoude_cv = """

Abdo Abou Jaoude, Ph.D.
Associate Professor
Department of Mathematics & Statistics
O: FNAS 0.12 (Main) & CA 114 B (Shouf Campus)
T: 09 218 950, Ext. 2545
E: aaboujaoude@ndu.edu.lb

Biography
Abdo Abou-Jaoudé has been teaching for more than 25 years and has a passion for educating and doing mathematics.
It gives him great pleasure to share his knowledge with others. He also believes that Mathematics is the gateway to
knowledge and to the deepest mysteries. This is what most philosophers across the ages have taught. By following in
the steps of the greatest mathematicians, he desires to be able to contribute throughout his career, even if modestly,
to this illustrious field of knowledge. He is extremely interested in the area of diagnostic and prognostic applied to
complex dynamic systems, which is the topic of his doctorate dissertation. His broad research interests encompass
the fields of pure and applied mathematics. He has published 23 international journal articles, 10 book chapters,
and six contributions to conference proceedings, in addition to 18 books on prognostics, mathematics, physics, and
computer science. He is also developing a new branch in Mathematics, that is, “The Complex Probability Paradigm”,
which joins Probability Theory with Complex Variables and Analysis.
Peer-reviewed Journals
International
• Abou Jaoude, A. (2020) The Paradigm of Complex Probability and The Central Limit Theorem, London Journal of
Research in Science: Natural and Formal (LJRS), London Journals Press, Vol. 20(5), pp: 1-57.
• Abou Jaoude, A. (2020) The Paradigm of Complex Probability and Prognostic Using FORM, London Journal of
Research in Science: Natural and Formal (LJRS), London Journals Press, Vol. 20(4), pp: 1-65.
• Abou Jaoude, A. (2019) The Paradigm of Complex Probability and Monte Carlo Methods, Systems Science and
Control Engineering (SSCE), Taylor and Francis Publishers, Vol. 7(1), pp: 407-451.
• Abou Jaoude, A. (2018) The Paradigm of Complex Probability and Ludwig Boltzmann's Entropy, Systems Science
and Control Engineering, Taylor and Francis Publishers, Vol. 6(1), pp: 108-149.
• Abou Jaoude, A. (2017) The Paradigm of Complex Probability and Analytic Nonlinear Prognostic for Unburied
Petrochemical Pipelines, Systems Science and Control Engineering, Taylor and Francis Publishers, Vol. 5(1), pp:
495-534.
• Abou Jaoude, A. (2017) The Paradigm of Complex Probability and Claude Shannon's Information Theory, Systems
Science and Control Engineering, Taylor and Francis Publishers, Vol. 5(1), pp: 380-425.
• Abou Jaoude, A. (2017) The Paradigm of Complex Probability and Analytic Linear Prognostic for Unburied
Petrochemical Pipelines, Systems Science and Control Engineering, Taylor and Francis Publishers, Vol. 5(1), pp:
178-214.
• Abou Jaoude, A. (2016) The Paradigm of Complex Probability and Analytic Nonlinear Prognostic for Vehicle Suspension
Systems, Systems Science and Control Engineering, Taylor and Francis Publishers, Vol. 4(1), pp: 334-378.
• Abou Jaoude, A. (2016) The Paradigm of Complex Probability and Chebyshev's Inequality, Systems Science and Control Engineering, Taylor and Francis Publishers, Vol. 4(1), pp: 99-137.
• Abou Jaoude, A. (2015) The Paradigm of Complex Probability and the Brownian Motion, Systems Science and Control Engineering, Taylor and Francis publishers, Vol. 3(1), pp: 478-503.
• Abou Jaoude, A. (2015) The Complex Probability Paradigm and Analytic Linear Prognostic for Vehicle Suspension Systems, American Journal of Engineering and Applied Sciences, Science Publications, Vol. 8(1), pp: 147-175.
• Abou Jaoude, A. (2015) Analytic and Linear Prognostic Model for A Vehicle Suspension System Subject to Fatigue, Systems Science and Control Engineering, Taylor and Francis publishers, Vol. 3(1), pp: 81-98.
• Abou Jaoude, A. (2014) Complex Probability Theory and Prognostic, Journal of Mathematics and Statistics, Science Publications, Vol. 10(1), pp: 1-24.
• Abou Jaoude, A. (2013) The Theory of Complex Probability and The First Order Reliability Method, Journal of Mathematics and Statistics, Science Publications, Vol. 9(4), pp: 310-324.
• Abou Jaoude, A. (2013) The Complex Statistics Paradigm and The Law of Large Numbers, Journal of Mathematics and Statistics, Science Publications, Vol. 9(4), pp: 289-304.
• Abou Jaoude, A. (2013) The Theory of Metarelativity: Beyond Albert Einstein's Relativity, Physics International, Science Publications, Vol. 4(2), pp: 97-109.
• El-Tawil, K., Abou Jaoude, A. (2013) Stochastic and Nonlinear Based Prognostic Model, Systems Science and Control Engineering, Taylor and Francis publishers, Vol. 1(1), pp: 66-81.
• Abou Jaoude, A., El-Tawil, K. (2013) Stochastic Prognostic Paradigm for Petrochemical Pipelines Subject to Fatigue, American Journal of Engineering and Applied Sciences, Science Publications, Vol. 6(2), pp: 145-160.
• Abou Jaoude, A., El-Tawil, K. (2013) Analytic and Nonlinear Prognostic for Vehicle Suspension Systems, American Journal of Engineering and Applied Sciences, Science Publications, Vol. 6(1), pp: 42-56.
• Abou Jaoude, A., Kadry, S., El-Tawil, K., Noura, H., Ouladsine, M. (2011) Analytic Prognostic for Petrochemical Pipelines, Journal of Mechanical Engineering Research, Vol. 3(3), pp: 64-74.
• Abou Jaoude, A., El-Tawil, K., Kadry, S., Noura, H., Ouladsine, M. (2010) Analytic Prognostic Model for A Dynamic System, International Review of Automatic Control, Vol. 3(6), pp: 568-577.
• Kadry, S., Kassem, H., Smaili, M., Abou Jaoude, A. (2010) Statistical Study of Complex Eigenvalues in Stochastic Systems, Research Journal of Applied Sciences, Engineering, and Technology, Vol. 2(3), pp: 233-238.
• Abou Jaoude, A., El-Tawil, K., Kadry, S. (2010) Prediction in Complex Dimension Using Kolmogorov's Set of Axioms, Journal of Mathematics and Statistics, Science Publications, Vol. 6(2), pp: 116-124.

Peer-reviewed Conference Proceedings
International
• Abou Jaoude, A., Noura, H., El-Tawil, K., Kadry, S., Ouladsine, M. (2012) Analytic Prognostic Model for Stochastic Fatigue of Petrochemical Pipelines, Proceedings of the Australian Control Conference, Sydney, Australia, pp: 233-240.
• Abou Jaoude, A., Noura, H., El-Tawil, K., Kadry, S., Ouladsine, M. (2012) Lifetime Analytic Prognostic for Petrochemical Pipes Subject to Fatigue, Proceedings of the 8th IFAC Symposium on Fault Detection, Supervision, and Safety of Technical Processes (Safeprocess), Mexico City, Mexico, pp: 707-713.
• Abou Jaoude, A., El-Tawil, K., Kadry, S., Noura, H., Ouladsine, M. (2011) Prognostic Model for Buried Tubes, Proceedings of the International Conference on Advanced Research and Applications in Mechanical Engineering, Notre Dame University-Louaize, Lebanon, pp: 169-174.
• El-Tawil, K., Abou Jaoude, A., Kadry, S., Noura, H., Ouladsine, M. (2010) Prognostic Based on Analytic Laws Applied to Petrochemical Pipelines, Proceedings of the International Conference on Computer-aided Manufacturing and Design, China, pp: 105-110.
• El-Tawil, K., Kadry, S., Abou Jaoude, A. (2009) Life Time Estimation Under Probabilistic Fatigue of Cracked Plates for Multiple Limits States, Proceedings of the Seventh International Conference of Numerical Analysis and Applied Mathematics, Crete, Greece, Vol. 2, pp: 1510-1514.
• Abou Jaoude, A., Kadry, S., El-Tawil, K. (2009) A Novel SFEM Algorithm Using the Probabilistic Transformation Method, Proceedings of the Seventh International Conference of Numerical Analysis and Applied Mathematics, Crete, Greece, Vol. 2, pp: 1505-1509.

Chapters in Books
International
• Abou Jaoude, A. (2024) The Paradigm of Complex Probability and Quantum Mechanics: The Infinite Potential Well Problem - The Position Wavefunction, Chapter 1, Part of the book: The Paradigm of Complex Probability and Quantum Mechanics, Book Publisher International, Kolkata: India.
• Abou Jaoude, A. (2024) The Paradigm of Complex Probability and Quantum Mechanics: The Infinite Potential Well Problem - The Momentum Wavefunction and The Wavefunction Entropies, Chapter 2, Part of the book: The Paradigm of Complex Probability and Quantum Mechanics, Book Publisher International, Kolkata: India.
• Abou Jaoude, A. (2024) The Paradigm of Complex Probability and Quantum Mechanics: The Quantum Harmonic Oscillator with Gaussian Initial Condition - The Position Wavefunction, Chapter 3, Part of the book: The Paradigm of Complex Probability and Quantum Mechanics, Book Publisher International, Kolkata: India.
• Abou Jaoude, A. (2024) The Paradigm of Complex Probability and Quantum Mechanics: The Quantum Harmonic Oscillator with Gaussian Initial Condition - The Momentum Wavefunction and The Wavefunction Entropies, Chapter 4, Part of the book: The Paradigm of Complex Probability and Quantum Mechanics, Book Publisher International, Kolkata: India.
• Abou Jaoude, A. (2024) The Paradigm of Complex Probability and Heisenberg's Quantum Uncertainty Principle, Chapter 5, Part of the book: The Paradigm of Complex Probability and Quantum Mechanics, Book Publisher International, Kolkata: India.
• Abou Jaoude, A. (2024) The Paradigm of Complex Probability and the Quantum Entropic Uncertainty Principle, Chapter 6, Part of the book: The Paradigm of Complex Probability and Quantum Mechanics, Book Publisher International, Kolkata: India.
• Abou Jaoude, A. (2024) The Paradigm of Complex Probability and the Weak and Strong Law of Large Numbers, Chapter 1, Part of the book: The Paradigm of Complex Probability, the Law of Large Numbers, and the Central Limit Theorem, Book Publisher International, Kolkata: India.
• Abou Jaoude, A. (2024) The Paradigm of Complex Probability and The Central Limit Theorem, Chapter 2, Part of the book: The Paradigm of Complex Probability, the Law of Large Numbers, and the Central Limit Theorem, Book Publisher International, Kolkata: India.
• Abou Jaoude, A. (2024) The Paradigm of Complex Probability and the Novel Dynamic Logic - The Model, Chapter 1, Part of the book: The Paradigm of Complex Probability, Prognostic, and Dynamic Logic, Book Publisher International, Kolkata: India.
• Abou Jaoude, A. (2024) The Paradigm of Complex Probability and the Novel Dynamic Logic - The Simulations, Chapter 2, Part of the book: The Paradigm of Complex Probability, Prognostic, and Dynamic Logic, Book Publisher International, Kolkata: India.
• Abou Jaoude, A. (2024) The Paradigm of Complex Probability and Analytic Nonlinear Prognostic for Unburied Petrochemical Pipelines - A Relation to Dynamic Logic, Chapter 3, Part of the book: The Paradigm of Complex Probability, Prognostic, and Dynamic Logic, Book Publisher International, Kolkata: India.
• Abou Jaoude, A. (2024) Fundamental Mathematical Methods and Concepts, Chapter 1, Part of the book: The Paradigm of Complex Probability and Markov Chains, Edition 1, Book Publisher International, Kolkata: India.
• Abou Jaoude, A. (2024) The Paradigm of Complex Probability and Markov Chains Transition Matrices, Chapter 2, Part of the book: The Paradigm of Complex Probability and Markov Chains, Edition 1, Book Publisher International, Kolkata: India.
• Abou Jaoude, A. (2024) The Paradigm of Complex Probability and Regular Markov Chains, Chapter 3, Part of the book: The Paradigm of Complex Probability and Markov Chains, Edition 1, Book Publisher International, Kolkata: India.
• Abou Jaoude, A. (2024) The Paradigm of Complex Probability and Absorbing Markov Chains, Chapter 4, Part of the book: The Paradigm of Complex Probability and Markov Chains, Edition 1, Book Publisher International, Kolkata: India.
• Abou Jaoude, A. (2023) The Paradigm of Complex Probability and Quantum Mechanics: The Quantum Harmonic Oscillator with Gaussian Initial Condition - The Momentum Wavefunction and The Wavefunction Entropies, Part of the book: Simulation Modeling - Recent Advances, New Perspectives and Applications, IntechOpen, London, United Kingdom.
• Abou Jaoude, A. (2023) The Paradigm of Complex Probability and Quantum Mechanics: The Quantum Harmonic Oscillator with Gaussian Initial Condition - The Position Wavefunction, Part of the book: Simulation Modeling - Recent Advances, New Perspectives, and Applications, IntechOpen, London, United Kingdom.
• Abou Jaoude, A. (2023) The Paradigm of Complex Probability and the Theory of Metarelativity - The General Model and Some Consequences of MCPP, Part of the book: Operator Theory - Recent Advances, New Perspectives and Applications, IntechOpen, London, United Kingdom.
• Abou Jaoude, A. (2023) The Paradigm of Complex Probability and the Theory of Metarelativity - A Simplified Model of MCPP, Part of the book: Operator Theory - Recent Advances, New Perspectives and Applications, IntechOpen, London, United Kingdom.
• Abou Jaoude, A. (2022) The Paradigm of Complex Probability and Quantum Mechanics: The Infinite Potential Well Problem - The Momentum Wavefunction and The Wavefunction Entropies, Part of the book: Applied Probability Theory - New Perspectives, Recent Advances and Trends, IntechOpen, London, United Kingdom.
• Abou Jaoude, A. (2022) The Paradigm of Complex Probability and Quantum Mechanics: The Infinite Potential Well Problem - The Position Wavefunction, Part of the book: Applied Probability Theory - New Perspectives, Recent Advances and Trends, IntechOpen, London, United Kingdom.
• Abou Jaoude, A. (2021) The Paradigm of Complex Probability and Isaac Newton's Classical Mechanics: On the Foundation of Statistical Physics, Part of the book: The Monte Carlo Methods - Recent Advances, New Perspectives and Applications, IntechOpen, London, United Kingdom.
• Abou Jaoude, A. (2021) The Paradigm of Complex Probability and Thomas Bayes' Theorem, Part of the book: The Monte Carlo Methods - Recent Advances, New Perspectives and Applications, IntechOpen, London, United Kingdom.
• Abou Jaoude, A. (2020) The Monte Carlo Techniques and The Complex Probability Paradigm, Part of the book: Forecasting in Mathematics - Recent Advances, New Perspectives and Applications, IntechOpen, London, United Kingdom.
• Abou Jaoude, A. (2020) Analytic Prognostic in the Linear Damage Case Applied to Buried Petrochemical Pipelines and the Complex Probability Paradigm, Part of the book: Fault Detection, Diagnosis and Prognosis, IntechOpen, London, United Kingdom.

Books
International
• Abou Jaoude, A. (2024) The Paradigm of Complex Probability and Metarelativity, Book Publisher International, Kolkata: India.
• Abou Jaoude, A. (2024) The Paradigm of Complex Probability and Quantum Mechanics, Book Publisher International, Kolkata: India.
• Abou Jaoude, A. (2024) The Paradigm of Complex Probability, the Law of Large Numbers, and the Central Limit Theorem, Book Publisher International, Kolkata: India.
• Abou Jaoude, A. (2024) The Paradigm of Complex Probability, Prognostic, and Dynamic Logic, Book Publisher International, Kolkata: India.
• Abou Jaoude, A. (2024) The Paradigm of Complex Probability, Numerical Analysis, and Chaos Theory, Book Publisher International, Kolkata: India.
• Abou Jaoude, A. (2024) The Paradigm of Complex Probability and Markov Chains, Edition 1, Book Publisher International, Kolkata: India.
• Abou Jaoude, A. (2024) The Framework of the Paradigm of Complex Probability and Monté Carlo Methods, Edition 1, Book Publisher International, Kolkata: India.
• Abou Jaoude, A. (2024) Recent Advances in Monte Carlo Methods, IntechOpen, London: United Kingdom.
• Abou Jaoude, A. (2024) Simulation Modeling - Recent Advances, New Perspectives, and Applications, IntechOpen, London: United Kingdom.
• Abou Jaoude, A. (2023) Applied Probability Theory - New Perspectives, Recent Advances and Trends, IntechOpen, London, United Kingdom.
• Abou Jaoude, A. (2023) Operator Theory - Recent Advances, New Perspectives and Applications, IntechOpen, London, United Kingdom.
• Abou Jaoude, A. (2022) The Monte Carlo Methods - Recent Advances, New Perspectives and Applications, IntechOpen, London, United Kingdom.
• Abou Jaoude, A. (2021) The Analysis of Selected Algorithms for the Statistical Paradigm, Volume 2, Generis Publishing, The Republic of Moldova.
• Abou Jaoude, A. (2021) The Analysis of Selected Algorithms for the Statistical Paradigm, Volume 1, Generis Publishing, The Republic of Moldova.
• Abou Jaoude, A. (2021) Forecasting in Mathematics - Recent Advances, New Perspectives and Applications, IntechOpen, London, United Kingdom.
• Abou Jaoude, A. (2019) The Analysis of Selected Algorithms for the Stochastic Paradigm, Cambridge Scholars Publishing, London, United Kingdom.
• Abou Jaoude, A. (2019) The Computer Simulation of Monté Carlo Methods and Random Phenomena, Cambridge Scholars Publishing, London, United Kingdom.
• Abou Jaoude, A. (2013) Automatic Control and Prognostic: Advanced Analytical Model for the Prognostic of Industrial Systems Subject to Fatigue, Scholars' Press, Saarbrucken, Germany.

Esteemed Indicators
• Editorial Board Member of Modern Intelligent Times, Innovationforever.
• Interview with the platform Faculti, London, United Kingdom, held on January, 24, 2023.
• Bircham International University Professor.
• Bircham International University Academic Board & Deans.
• Bircham International University Board of Trustees.
• Education Quality Accreditation Commission (EQAC) Member.
• Cambridge Scholars Publishing Editorial Advisory Board, London, United Kingdom.

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

    entities = ner(cv)

    print(f"professor name: {name_professor}")

    print(f"professor email: {email_professor}")

    skills = {ent['word'] for ent in entities if ent['entity_group']
              in {"TECHNICAL", "TECHNOLOGY", "BUSINESS", "SOFT"}}
    print("Extracted Skills:")
    for skill in skills:
        print("-", skill)


def func_course(course):
    doc_course = nlp(course)
    course_id = "failed"
    for ent in doc_course.ents:
        if ent.label_ == "COURSE_ID":
            course_id = ent.text
            break

    start_idx = course.find(f"- {course_id}")
    course_name = course[:start_idx].strip().rstrip("-").strip()
    print(f"Course ID: {course_id}")
    print(f"Course Name: {course_name}")


print("CV: \n")
func_cv(history_cv)
print("Course Description: \n")
func_course(course_cs)
