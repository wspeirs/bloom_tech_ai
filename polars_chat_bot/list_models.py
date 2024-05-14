import boto3

client = boto3.client('bedrock')

res = client.list_foundation_models(byProvider='amazon')
summaries = res['modelSummaries']

print(res)

for summary in summaries:
    print(f"{summary['modelId']}")
    print(f"\tARN: {summary['modelArn']}\t")
    print(f"\tName: {summary['modelName']}\t")
    print(f"\tinputs: {summary['inputModalities']}\t")
    print(f"\toutputs: {summary['outputModalities']}\t")

