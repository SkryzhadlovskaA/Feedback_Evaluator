"""
Erasmus+ outcome categories and per-project topic definitions.

Each category has a label (used in output), description, and example sentences
for embedding-based matching in learningoutcomes_detector.py.
"""

ERASMUS_OUTCOMES = {
    "learning_performance": {
        "label": "Improved learning performance",
        "description": "better learning performance, improved understanding, stronger ability to learn effectively",
        "examples": [
            "I learned better during this project",
            "I improved how I learn",
            "I understand topics more effectively now"
        ]
    },
    "employability": {
        "label": "Enhanced employability and improved career prospects",
        "description": "better employability, career prospects, professional readiness, job-related skills",
        "examples": [
            "This experience will help me in my future job",
            "I gained skills useful for my career",
            "I feel more prepared professionally"
        ]
    },
    "initiative_entrepreneurship": {
        "label": "Increased sense of initiative and entrepreneurship",
        "description": "taking initiative, being proactive, entrepreneurial mindset, leadership, starting things independently",
        "examples": [
            "I became more proactive",
            "I learned to take initiative",
            "I feel more able to start my own ideas"
        ]
    },
   "self_empowerment": {
    "label": "Increased self-empowerment and self-esteem",
    "description": "greater self-confidence, self-esteem, feeling empowered, personal confidence, belief in oneself, realizing one's own abilities and potential",
    "examples": [
        "I became more confident",
        "I believe in myself more now",
        "I feel more empowered",
        "I learned that I am capable of more than I thought",
        "I am less afraid to share my opinion"
    ]
},
    "language_digital": {
        "label": "Improved foreign language and digital competences",
        "description": "better foreign language skills, communication in another language, digital competence, technical skills",
        "examples": [
            "My English improved",
            "I practiced speaking another language",
            "I improved my digital skills"
        ]
    },
    "intercultural_awareness": {
        "label": "Enhanced intercultural awareness",
        "description": "better understanding of other cultures, intercultural learning, cultural openness, openness to diversity",
        "examples": [
            "I understand other cultures better",
            "I became more open-minded toward different cultures",
            "I learned about cultural differences"
        ]
    },
    "active_participation": {
        "label": "More active participation in society",
        "description": "active citizenship, civic engagement, participation in society, social involvement",
        "examples": [
            "I want to participate more in my community",
            "I became more active in society",
            "I feel more engaged as a citizen"
        ]
    },
    "positive_interactions": {
    "label": "Enhanced positive interactions with people from different backgrounds",
    "description": "better collaboration, teamwork, communication with others, positive interaction, working with diverse people, inclusion and mutual respect",
    "examples": [
        "I learned to work with people from different backgrounds",
        "I communicate better with diverse groups",
        "I became better at interacting with different people",
        "I improved my communication and teamwork skills",
        "I learned to collaborate better with others"
    ]
},
    "eu_values_awareness": {
        "label": "Better awareness of the European project and the EU values",
        "description": "understanding European values, EU values, European project, European identity",
        "examples": [
            "I understand EU values better",
            "I learned more about Europe",
            "I became more aware of the European project"
        ]
    },
    "future_motivation": {
        "label": "Increased motivation for future education or training",
        "description": "greater motivation to continue learning, future education, training, continued self-development",
        "examples": [
            "I want to join more trainings in the future",
            "I feel motivated to keep learning",
            "This inspired me to continue my education"
        ]
    }
}

OTHER_OUTCOME_LABEL = "Other"

# Add one entry per project_id before running the pipeline on a new project.
PROJECT_TOPICS = {
    "Nature_Training_Course": {
        "label": "Project topic: Sustainability and environmental awareness",
        "description": (
            "sustainability, sustainable living, eco-friendly practices, environmental impact, "
            "ecology, nature, contamination, green habits, environmental education"
        ),
        "examples": [
            "How to live life with sustainability",
            "How I can live sustainably with the focus on the environment",
            "About ecology in different countries",
            "I learned new practices for CO2 reduction",
            "More knowledge about critical thinking and connection with nature"
        ]
    },
    "Career_Youth_Exchange": {
        "label": "Project topic: Career development and professional skills",
        "description": (
            "career development, employability, professional skills, CV preparation, "
            "public speaking, leadership, teamwork, success, job opportunities, workplace skills"
        ),
        "examples": [
            "Teamwork, public speaking and time management",
            "yes, i understood there are many career opportunities",
            "Developed leadership skills",
            "Yes, my biggest improvement would be public speaking",
            "The CV preparation activity helped me a lot"
        ]
    }
}


def get_outcome_labels_for_project(project_id: str | None) -> list[str]:
    """All labels used for matching: 10 Erasmus + project topic (if defined)."""
    labels = [item["label"] for item in ERASMUS_OUTCOMES.values()]
    if project_id and project_id in PROJECT_TOPICS:
        labels.append(PROJECT_TOPICS[project_id]["label"])
    labels.append(OTHER_OUTCOME_LABEL)
    return labels