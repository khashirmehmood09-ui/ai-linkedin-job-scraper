import gradio as gr
import requests
from bs4 import BeautifulSoup
import pandas as pd 
from datetime import datetime
import time
from urllib.parse import quote
from flair.models import SequenceTagger
from flair.data import Sentence


import threading
import os 
import random
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

flair_model = SequenceTagger.load("kaliani/flair-ner-skill")

experience_level_mapping={
    "Internship":"f_E=1",
     "Entrylevel":"f_E=2",
    "Associate":"f_E=3",
    "Mid_Senior level":"f_E=4"
}
work_type_mapping={
    "On_site":"f_WT=1",
    "Hybrid":"f_WT=2",
    "Remote":"f_WT=3"
}

time_filter_mapping={
    "Past 24 hours":"f_TPR=r86400",
    "Past week":"f_TPR=r604800",
    "Past month":"f_TPR=r2592000",
}

experience_level_mapping={
    "Internship":"f_E=1",
     "Entrylevel":"f_E=2",
    "Associate":"f_E=3",
    "Mid_Senior level":"f_E=4"
}
work_type_mapping={
    "On_site":"f_WT=1",
    "Hybrid":"f_WT=2",
    "Remote":"f_WT=3"
}

time_filter_mapping={
    "Past 24 hours":"f_TPR=r86400",
    "Past week":"f_TPR=r604800",
    "Past month":"f_TPR=r2592000",
}


KNOWN_SKILLS = [
    "Python","SQL","PostgreSQL","MySQL","MongoDB","SQLite",
    "React","React JS","Next.js","Node.js","Express.js",
    "FastAPI","Flask","Django",
    "AWS","Azure","GCP",
    "Docker","Kubernetes",
    "Terraform",
    "Snowflake",
    "Redis",
    "RabbitMQ",
    "Celery",
    "Playwright",
    "Selenium",
    "BeautifulSoup",
    "Pandas",
    "NumPy",
    "Scikit-learn",
    "TensorFlow",
    "PyTorch",
    "LLM",
    "Claude",
    "Cursor",
    "Git",
    "GitHub",
    "CI/CD",
    "JWT",
    "RBAC",
    "CloudWatch",
    "RDS",
    "S3"
]

description = """
Devsinc is seeking a Senior AI/Machine Learning Engineer with a strong background in Python, machine learning, and deep learning frameworks. Our ideal candidate will excel in cloud environments (such as AWS, Azure, GCP, and Databricks), possess the ability to design innovative AI solutions, exhibit expertise in Python, Spark, and relevant AI libraries, and demonstrate a high level of personal responsibility.

Principal Responsibilities:

Develop and maintain scalable, secure AI and machine learning applications utilizing Python, machine learning frameworks (e.g., TensorFlow, PyTorch), and cloud services.

Design and implement machine learning models and algorithms to support various AI-driven client applications, with a focus on user interface interactions and AI-driven features.

Integrate third-party AI/ML APIs and services into existing web applications.

Promote a data-driven and machine learning approach with a commitment to delivering valuable AI enhancements consistently.

Deep understanding of LLMs (open source). We focus on a wide variety of NLP use cases including writing assistance, summarization, and concept extraction.

Lead and participate in NLP and computer vision model development, providing constructive feedback to foster a culture of continuous improvement among team members.

Requirements:

3–6 years of experience as an active coder, with proficiency in Python 3.x, strong Object-Oriented Programming (OOP) skills, and familiarity with modern Python features.

Proven experience in Natural Language Processing (NLP) and Computer Vision (CV).

In-depth knowledge of essential Python libraries such as NumPy, Pandas, Scikit-learn, TensorFlow, PyTorch, Keras, Transformers, and others relevant to machine learning.

Competence in working with cloud environments (AWS, Azure, GCP, Databricks) and Linux, including Lambda/Serverless, SQS, SNS, S3, and EC2.

Experience deploying Transformer-based models into production.

Proficiency in Django or Flask is a huge plus.

Strong expertise in source control, code review, and repository management using Git.

Familiarity with software engineering principles and design patterns, including Dependency Injection, SOLID, Service Containers, and Providers.

Experience with containerization technologies like Docker.

Proficiency in building highly distributed, eventually consistent AI systems.

Familiarity with microservices architecture and message broker systems.

Expertise in various machine learning testing methodologies, including unit testing, integration testing, performance testing, and load testing.

Knowledge of data visualization, monitoring, and alerting concepts along with relevant tooling.

Excellent knowledge of Relational Databases, SQL, and ORM technologies such as SQLAlchemy.

Knowledge of LLMs, including fine-tuning and deployment integration with web applications.
"""

def get_skills(text):
    sentence = Sentence(text)
    flair_model.predict(sentence)

    skills = [entity.text for entity in sentence.get_spans("ner")]

    unique_skills = []
    for skill in skills:
        if skill not in unique_skills:
            unique_skills.append(skill)

    return unique_skills

get_skills(description)


class ScraperManager:

    def __init__(self):
        self.stop_event = threading.Event()
        self.current_df = pd.DataFrame()
        self.lock = threading.Lock()

    def reset(self):
        self.stop_event.clear()
        self.current_df = pd.DataFrame()

    def add_job(self, job_data):
        with self.lock:
            new_df = pd.DataFrame([job_data])
            self.current_df = pd.concat(
                [self.current_df, new_df],
                ignore_index=True
            )


scraper_manager = ScraperManager()

def save_csv(df, filename="jobs"):
    try:
        os.makedirs("saved_jobs", exist_ok=True)

        if not filename or filename.strip() == "":
            filename = f"jobs_{int(time.time())}"

        full_path = f"saved_jobs/{filename}.csv"

        df.to_csv(full_path, index=False)

        return full_path

    except Exception as e:
        print(f"Save error: {str(e)}")
        return None


def process_job(job, work_type, exp_level, position):
    try:
        # find job title
        title_element = job.find('h3', class_='base-search-card__title')

        # find company name
        company_element = job.find('a', class_='hidden-nested-link')

        # find location
        loc_element = job.find('span', class_='job-search-card__location')

        # find job link
        link_element = job.find('a', class_='base-card__full-link')

        # check all data exists
        if not all([title_element, company_element, loc_element, link_element]):
            return None

        # clean text
        title = title_element.text.strip()
        company = company_element.text.strip()
        loc = loc_element.text.strip()

        # clean link
        link = link_element['href'].split('?')[0]


        # setup web session with retries
        session = requests.Session()

        retries = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )

        session.mount(
            'https://',
            HTTPAdapter(max_retries=retries)
        )


        # Default values
        desc = "Description not available"
        skills = []


        try:
            # wait to avoid blocking
            time.sleep(random.uniform(2, 5))

            # fetch job page
            response = session.get(
                link,
                headers={
                    'User-Agent': random.choice([
                        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/91 Safari/537.36',
                        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Safari/605.1.15',
                        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/92 Safari/537.36'
                    ]),
                    'Accept-Language': 'en-US,en;q=0.9',
                },
                timeout=10
            )


            # parse HTML
            job_soup = BeautifulSoup(
                response.text,
                'html.parser'
            )


            description_selectors = [
                'div.description__text',
                'div.show-more-less-html__markup',
                'div.core-section-container__content',
                'section.core-section-container'
            ]


            for selector in description_selectors:

                desc_element = job_soup.select_one(selector)

                if desc_element:
                    desc = desc_element.get_text(
                        '\n'
                    ).strip()

                    # AI skill extraction
                    skills = get_skills(desc)

                    break


        except Exception as e:
            print(f"Error processing description {link}: {str(e)}")


        # return job details
        return {
            "Position": position,
            "Date": datetime.now().strftime('%Y-%m-%d'),
            "Work type": work_type,
            "Level": exp_level,
            "Title": title,
            "Company": company,
            "Location": loc,
            "Link": f"[{link}]({link})",
            "Description": desc,
            "Skills": ", ".join(skills[:5]) if skills else "No skills detected"
        }


    except Exception as e:
        print(f"Error processing job card: {str(e)}")
        return None


def scrape_jobs(location, position, work_types, exp_levels, time_filter):

    session = requests.Session()

    retries = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504]
    )

    session.mount(
        "https://",
        HTTPAdapter(max_retries=retries)
    )


    encoded_position = quote(position)
    encoded_location = quote(location)


    for work_type in work_types:

        for exp_level in exp_levels:

            if scraper_manager.stop_event.is_set():
                return


            try:

                base_url = (
                    "https://www.linkedin.com/jobs/search/?"
                    f"keywords={encoded_position}"
                    f"&location={encoded_location}"
                    f"&{work_type_mapping[work_type]}"
                    f"&{experience_level_mapping[exp_level]}"
                    f"&{time_filter_mapping[time_filter]}"
                    "&radius=0"
                )


                print("\nSearching:", position)
                print("URL:", base_url)


                response = session.get(
                    base_url,
                    headers={
                        "User-Agent": random.choice([
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
                        ]),
                        "Accept-Language": "en-US,en;q=0.9"
                    },
                    timeout=10
                )


                soup = BeautifulSoup(
                    response.text,
                    "html.parser"
                )


                try:

                    count = soup.find(
                        "span",
                        class_="results-context-header__job-count"
                    )

                    total_jobs = int(
                        count.text.replace(",", "")
                    )

                except:

                    total_jobs = 25


                total_jobs = min(total_jobs, 100)


                for start in range(0, total_jobs, 25):

                    if scraper_manager.stop_event.is_set():
                        return


                    time.sleep(
                        random.uniform(2,5)
                    )


                    page_url = (
                        f"{base_url}"
                        f"&start={start}"
                    )


                    try:

                        response = session.get(
                            page_url,
                            timeout=10,
                            headers={
                                "User-Agent": "Mozilla/5.0"
                            }
                        )


                        soup = BeautifulSoup(
                            response.text,
                            "html.parser"
                        )


                        jobs = soup.find_all(
                            "div",
                            class_="base-card"
                        )


                        print(
                            "Jobs found:",
                            len(jobs)
                        )


                    except Exception as e:

                        print(
                            "Page error:",
                            e
                        )

                        continue



                    for job in jobs:


                        if scraper_manager.stop_event.is_set():
                            return


                        data = process_job(
                            job,
                            work_type,
                            exp_level,
                            position
                        )


                        if data:

                            scraper_manager.add_job(
                                data
                            )

                            yield data



            except Exception as e:

                print(
                    "Scraper error:",
                    e
                )



# Define function to start scraping
print("run_scraper cell started")
def run_scraper(cities, states, positions, work_types, exp_levels, time_filter):

    # Clear old data
    scraper_manager.reset()

    # Split cities into list
    cities_list = [c.strip() for c in cities.split(",") if c.strip()]

    # Split states into list
    states_list = [s.strip() for s in states.split(",") if s.strip()]

    # Combine cities and states
    locations = [
        f"{city}, {state}"
        for city in cities_list
        for state in states_list
    ]

    # Clean positions
    positions_list = [
    p.strip()
    for p in positions.split(",")
    if p.strip()
]

    # Background worker
    def worker():

        for loc in locations:

            for pos in positions_list:

                if scraper_manager.stop_event.is_set():
                    return

                # Call scraper
                for _ in scrape_jobs(
                    loc,
                    pos,
                    work_types,
                    exp_levels,
                    time_filter,
                ):
                    pass

    # Start background thread
    thread = threading.Thread(target=worker)
    thread.start()

    # Update UI while scraping
    while thread.is_alive():

        time.sleep(0.5)

        with scraper_manager.lock:
            yield (
                "Scraping in progress...",
                scraper_manager.current_df
            )

    # Final message
    if scraper_manager.stop_event.is_set():
        yield (
            "Scraping stopped!",
            scraper_manager.current_df
        )
    else:
        yield (
            "Scraping completed!",
            scraper_manager.current_df
        )
print("run_scraper defined")


import gradio as gr

with gr.Blocks(
    title="AI-Powered LinkedIn Job Scraper"
) as app:

    gr.Markdown("""
    <h1 style="
        text-align:center;
        color:#00ffff;
        font-size:42px;
        text-shadow:
        0 0 5px #00ffff,
        0 0 15px #00ffff,
        0 0 30px #008cff;
    ">
    ⚡ AI-POWERED LINKEDIN JOB SCRAPER ⚡
    </h1>

    <p style="
        text-align:center;
        color:#39ff14;
        font-size:18px;
    ">
    AI Skills Extraction | Real-Time Job Scraping | Automated CSV Export
    </p>
    """)


    with gr.Row():

        with gr.Column(scale=1):

            cities = gr.Textbox(
                label="🌍 Cities (comma-separated)"
            )


            states = gr.Textbox(
                label="📍 States/Countries (comma-separated)"
            )


            positions = gr.Textbox(
                label="💼 Positions (comma-separated)"
            )


            work_types = gr.CheckboxGroup(
                choices=list(work_type_mapping.keys()),
                label="🏢 Work Types"
            )


            exp_levels = gr.CheckboxGroup(
                choices=list(experience_level_mapping.keys()),
                label="📈 Experience Levels"
            )


            time_filter = gr.Dropdown(
                choices=list(time_filter_mapping.keys()),
                label="⏳ Time Filter"
            )



            with gr.Row():

                start_btn = gr.Button(
                    "🚀 Start Scraping",
                    variant="primary"
                )


                stop_btn = gr.Button(
                    "⛔ Stop",
                    variant="stop"
                )



            status = gr.Textbox(
                label="📡 Status"
            )



            results = gr.DataFrame(

                headers=[
                    "Position",
                    "Date",
                    "Work Type",
                    "Level",
                    "Title",
                    "Company",
                    "Location",
                    "Link",
                    "Skills"
                ],

                datatype=[
                    "str",
                    "str",
                    "str",
                    "str",
                    "str",
                    "str",
                    "str",
                    "markdown",
                    "str"
                ],

                interactive=False
            )



            with gr.Row():

                filename = gr.Textbox(
                    label="📁 Filename",
                    placeholder="my_jobs"
                )


                save_btn = gr.Button(
                    "💾 Save & Download CSV",
                    variant="secondary"
                )



            download_file = gr.File(
                label="⬇️ Download Your CSV"
            )



    # Start scraping

    start_btn.click(

        fn=run_scraper,

        inputs=[
            cities,
            states,
            positions,
            work_types,
            exp_levels,
            time_filter
        ],

        outputs=[
            status,
            results
        ]

    )



    # Stop scraping

    stop_btn.click(

        fn=lambda: scraper_manager.stop_event.set(),

        outputs=[]

    )



    # Save CSV

    save_btn.click(

        fn=save_csv,

        inputs=[
            results,
            filename
        ],

        outputs=download_file

    )



if __name__ == "__main__":

    app.launch(
        theme=gr.themes.Base(
            primary_hue="cyan",
            secondary_hue="blue"
        ),
        css="""

        body {
            background: #050816;
        }

        .gradio-container {
            background: #050816;
        }

        h1 {
            font-family: Arial, sans-serif;
            letter-spacing: 2px;
        }

        """,
        share=True
    )