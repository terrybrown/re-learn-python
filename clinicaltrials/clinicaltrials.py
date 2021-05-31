
def request_trials(term):

    def clinical_trials_url = f"https://clinicaltrials.gov/api/query/full_studies?expr={term}&min_rnk=1&max_rnk=100&fmt=json"

    