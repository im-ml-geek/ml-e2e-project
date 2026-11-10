# ml-e2e-project
## Course Lead Conversion Prediction

A machine learning project that predicts whether a course lead will convert. The admissions team can use the predicted probability to prioritise follow-up. The API returns the probability and flags a lead when it reaches the selected cutoff of 0.40.

## Data and model

The ML Zoomcamp course lead scoring dataset contains 5,000 leads. The target is converted: 1 means the lead converted and 0 means they did not. Overall, 57.66% of leads converted.

The project uses seven inputs: annual income, number of courses viewed, interaction count, lead source, industry, employment status, and location. It excludes lead_score because its availability before conversion was not verified.

Exploratory analysis: Checked missing values, target balance, and conversion rates by lead source. For example, 75.91% of referral leads converted in the training set, compared with 42.98% of paid-ad leads. These are observed rates, not evidence of causation.

Data split: Used a stratified 60% training / 20% validation / 20% test split (3,000 / 1,000 / 1,000 leads). Stratification kept the class proportions similar across the three sets.

Preprocessing: Filled missing numerical values with the training median, scaled numerical features for logistic regression, filled missing categorical values with a placeholder, and one-hot encoded categorical features. The saved pipeline applies the same transformations when serving predictions.

Model selection: Compared a dummy baseline, tuned logistic regression, and tuned random forest on validation ROC AUC. Selected logistic regression with C=1. A cutoff of 0.40

## Model comparison

| Model | Validation ROC AUC |
| --- | ---: |
| Dummy baseline | 0.5000 |
| Logistic regression | **0.8043** |
| Random forest | 0.8014 |

## Model performance

| Metric | Test score |
| --- | ---: |
| ROC AUC | 0.776 |
| Precision | 0.689 |
| Recall | 0.872 |
| F1 score | 0.770 |

At the 0.40 cutoff, the model flagged 730 of 1,000 test leads and caught 503 of 577 actual converters. It also flagged 227 leads who did not convert. A team using this cutoff would need enough capacity to review the flagged leads.

## Test confusion matrix

| Actual outcome | Not flagged | Flagged |
| --- | ---: | ---: |
| Did not convert | 196 | 227 |
| Converted | 74 | 503 |

## Project files

| File | Purpose |
| --- | --- |
| Project notebook (`.ipynb`) | Explores the data, compares models, and evaluates performance |
| `train.py` | Trains and evaluates the model, then saves the fitted pipeline and cutoff |
| `predict.py` | Serves predictions through the FastAPI `/predict` endpoint |
| `requirements.txt` | Lists the Python dependencies |
| `Dockerfile` | Defines how to build the API image |
| `lead_conversion_model.joblib` | Stores the fitted pipeline and cutoff used by the API |

## Run locally
Run these commands from the project root, where train.py and requirements.txt are located. Use the Python version specified in the Dockerfile to keep the local and container environments aligned.

1. Create a virtual environment and install dependencies.
   
   `python3 -m venv .venv`
   `source .venv/bin/activate`
   `python -m pip install -r requirements.txt`

On Windows PowerShell, activate the environment with `\.venv\Scripts\Activate.ps1` instead of the source command.

2. Download the data and train the model.
   
`curl -fL https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/course_lead_scoring_2026.csv -o course_lead_scoring_2026.csv`
`python train.py`

The training script prints the test results and generates lead_conversion_model.joblib. To reproduce the exploration and model comparison, open the project notebook from the project root and run all cells in order using the project environment.

3. Start the API.
   
`python -m uvicorn predict:app --host 127.0.0.1 --port 9696`

Open local API docs : http://127.0.0.1:9696/docs or use the example request below. Press Ctrl+C to stop the server.

## Run with Docker
Start Docker, then run these commands from the project root. Run the training step first if lead_conversion_model.joblib is missing; the Docker build needs that file.
`docker build -t lead-conversion-api .` 
` docker run --rm -p 127.0.0.1:9697:9696 lead-conversion-api`

Docker forwards port 9697 on your computer to port 9696 inside the container. Open the container API docs to try the endpoint. Press Ctrl+C to stop the container. If you retrain the model, rebuild the image so it includes the new artifact.

## Test a prediction
While the container is running, open a second terminal:

`curl -fsS http://127.0.0.1:9697/predict \
  -H 'Content-Type: application/json' \
  --data '{
    "annual_income": null,
    "number_of_courses_viewed": 3,
    "interaction_count": 5,
    "lead_source": "referral",
    "industry": null,
    "employment_status": null,
    "location": null
  }'`

  For the local Python server, change the port to 9696. The example returned:

  `{
  "conversion_probability": 0.9177006935012685,
  "flag_for_follow_up": true
  }`

  The probability exceeds 0.40, so the API flags the lead. A successful check verifies both the HTTP status (200 OK) and the response values.

## Cloud deployment

The lead scoring API is deployed to Google Cloud Run in `europe-west2`.
The Docker image is stored in Google Artifact Registry.

- **Live demo:** [Open the interactive API docs](https://lead-conversion-api-899909596425.europe-west2.run.app/docs)
- **Prediction endpoint:** `POST /predict`
- **Verification:** Cloud Run logs recorded successful prediction requests with HTTP `200 OK`.

### Redeploy the API

Run these commands from the project root with Docker running and the
Google Cloud CLI installed. The `ml-images` repository already exists.
Run `python train.py` first if `lead_conversion_model.joblib` is missing.

```bash
gcloud auth login
gcloud auth configure-docker europe-west2-docker.pkg.dev

IMAGE="europe-west2-docker.pkg.dev/lead-conversion-demo/ml-images/lead-conversion-api:v1"

docker build --platform linux/amd64 -t "$IMAGE" .
docker push "$IMAGE"

gcloud run deploy lead-conversion-api \
  --image "$IMAGE" \
  --project lead-conversion-demo \
  --region europe-west2 \
  --port 9696 \
  --no-invoker-iam-check
```

### Test the cloud endpoint

Open the live API docs, expand **POST `/predict`**, and click **Try it out**.
Use the sample JSON from **Test a prediction** above and click **Execute**.
Expect HTTP `200`, a conversion probability of approximately `0.9177`, and
`"flag_for_follow_up": true`.
