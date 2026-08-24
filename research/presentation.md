# MLOPs in practical applications - presentation notes

## Main requirements:

- A 1.5-hour presentation
- Target audience: EE (electrical engineering) and mathematics students
- An FFT (Fourier transform) demo that runs throughout the entire presentation
- Important papers (Zhou et al. 2019 "Hidden Technical Debt", MLTest, latest papers)
- A security section toward the end
- Distinction between industry and students
- Start with an explanation of DevOps (code vs. model due to data drift)
- Goal of a perfect 1.0 (in the German grading system)


A demo will be the guiding line throughout the whole presentation.  
Demo: FFT machine learning model

Data for the model is not yet present but will be synthetically generated. For first tests some small dataset can already be generated.


## Thoughts about Presentation Structure:
1. explaining DevOps briefly
    - **Goal of DevOPs:** automating the Software Development Life Cycle (SDLC)
    - add following in some way:
    "In DevOps, if you don't change the code, the software stays the same.
    In ML, even if your **code stays exactly the same**, your **model can become useless** because the **data**
    changed (e.g., consumer behavior changed during a pandemic). This is something DevOps is not designed to catch."
    - ML itroduces another variable: **The Data**

2. MLOPs explanation
    - beginning with Demo
    - showing the data
    - showcasing how to train a model in a jupyter notebook (or finetune if full training would take too long)
    - going into - The crisis of ML - why do we need MLOPs - Problem describtion eith the paper: "Hidden Technical Debt in Machine Learning Systems.
3. The core -techincal parts 
    - showcasing all with the demo
	- version control not just for code, but also for data
	- tracking of experiments - saving weights and biases
	- Automated Pipelines
		- testing and validating 
		- model deployment
		- auto retrain
	- model monitoring and drift
		- concept drift: input and output changes (e.g. customer behaviour change)
		- data drift: data distribution changes
4. Common used tools in MLOPs
    - going into github with the demo model -> CI/CD, github actions, etc.
5. The full MLOPs flow
6. MLOPs for personal projects vs industry  # extra point or should i build this into the other points?
	- personal projects: GitHub, Google Colab/Kaggle, Streamlit, Docker
	- industry: focus also on Scalability, Reliability, and Compliance, Security ;  Kubernetes, Kubeflow, AWS SageMaker/Azure ML, Terraform, weights and biases, Tecton, Feast, Airflow
6. Also having an eye on security in MLOPs (max 10 minutes long)
    - quick thoughts on security

**Ideas that might be included**

Problem: "Accuracy is 95%, right?"  
Problem: "So which run was the good one?"  
Problem: "I'm getting different results"  
Problem: "model_final_FINAL2.pkl"  
Problem: "Someone should be able to use this"  