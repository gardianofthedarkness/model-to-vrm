"""The outside models the pipeline calls, named in one place (change a model here, not in the step scripts)."""
JEV_MODEL = 'jev-1.14.0'                      # the decision model (boxes, choices, scores)
JEV_ENDPOINT = 'https://api.typesafe.ai/v1/systemone'
SAM3_MODEL = 'facebook/sam3'                  # text-prompted instance masks
