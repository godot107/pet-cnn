"""Run `train` as a SageMaker training job on a spot GPU: the AWS replacement for the Azure ML
compute that ran out of memory in 2023. boto3 only (no SageMaker SDK).

Flow: upload data/processed/ to S3 once -> tar src + config as the job's source dir -> create the
job (PyTorch Deep Learning Container runs sm_entry.py) -> `sm-fetch` pulls model.tar.gz into models/.
"""
from __future__ import annotations

import io
import json
import tarfile
import time

from . import config, data

TAGS = [{"Key": k, "Value": v} for k, v in {
    "project": "pet-cnn", "environment": "dev", "owner": "williemaize", "cost-center": "personal",
    "managed-by": "petcnn-sagemaker", "component": "lab", "repository": "github.com/godot107/pet-cnn"}.items()]


def _session():
    import boto3
    return boto3.session.Session(region_name=config.load()["sagemaker"]["region"])


def _stack_outputs(sess) -> dict:
    stack = config.load()["sagemaker"]["lab_stack"]
    cf = sess.client("cloudformation")
    try:
        outs = cf.describe_stacks(StackName=stack)["Stacks"][0]["Outputs"]
    except cf.exceptions.ClientError as e:
        raise SystemExit(f"Lab stack {stack} not found ({e}). Run deploy/deploy_lab.sh first.") from e
    return {o["OutputKey"]: o["OutputValue"] for o in outs}


def preflight(max_usd: float | None = None) -> float:
    """Worst-case job cost (full max runtime at on-demand price). Refuses above max_usd."""
    cfg = config.load()
    sm, prices = cfg["sagemaker"], cfg["prices"]
    rate = prices[sm["instance_type"]]
    worst = rate * sm["max_runtime_min"] / 60
    limit = max_usd if max_usd is not None else sm["max_usd"]
    print(f"preflight: {sm['instance_type']} at ${rate}/h x {sm['max_runtime_min']} min max = ${worst:.2f} worst case "
          f"(spot is usually 60-70% less); limit ${limit:.2f}")
    if not prices.get("verified"):
        print("  WARNING: prices.verified is false in config.yaml; check https://aws.amazon.com/sagemaker/pricing/")
    if not sm.get("image_verified"):
        print("  WARNING: sagemaker.image_verified is false; check the DLC tag exists (see config.yaml)")
    if worst > limit:
        raise SystemExit(f"refusing: worst case ${worst:.2f} > ${limit:.2f}")
    return worst


def _source_tar() -> bytes:
    """sourcedir.tar.gz: sm_entry.py + requirements.txt at the root, plus config.yaml and src/petcnn."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        t.add(config.ROOT / "sagemaker" / "sm_entry.py", "sm_entry.py")
        t.add(config.ROOT / "sagemaker" / "requirements.txt", "requirements.txt")
        t.add(config.ROOT / "config.yaml", "config.yaml")
        t.add(config.ROOT / "src" / "petcnn", "src/petcnn",
              filter=lambda ti: None if "__pycache__" in ti.name else ti)
    return buf.getvalue()


def _upload_data(s3, bucket: str) -> str:
    prefix = "data/processed/"
    for name in (data.CACHE, "index.csv"):
        path = config.PROCESSED / name
        try:
            head = s3.head_object(Bucket=bucket, Key=prefix + name)
            if head["ContentLength"] == path.stat().st_size:
                continue
        except s3.exceptions.ClientError:
            pass
        print(f"uploading {name} ({path.stat().st_size / 1e6:.0f} MB)")
        s3.upload_file(str(path), bucket, prefix + name)
    return f"s3://{bucket}/{prefix}"


def launch(arch: str, scheme: str, name: str, task: str | None = None, epochs: int | None = None,
           max_usd: float | None = None, wait: bool = True) -> str:
    preflight(max_usd)
    sm_cfg = config.load()["sagemaker"]
    sess = _session()
    outs = _stack_outputs(sess)
    bucket, role = outs["Bucket"], outs["RoleArn"]
    s3 = sess.client("s3")
    data_uri = _upload_data(s3, bucket)

    job = f"pet-cnn-{name}-{time.strftime('%Y%m%d-%H%M%S')}"[:63]
    code_key = f"code/{job}/sourcedir.tar.gz"
    s3.put_object(Bucket=bucket, Key=code_key, Body=_source_tar())

    hp = {"sagemaker_program": "sm_entry.py", "sagemaker_submit_directory": f"s3://{bucket}/{code_key}",
          "sagemaker_region": sess.region_name, "arch": arch, "scheme": scheme, "name": name}
    if task:
        hp["task"] = task
    if epochs:
        hp["epochs"] = epochs
    stopping = {"MaxRuntimeInSeconds": sm_cfg["max_runtime_min"] * 60}
    if sm_cfg["spot"]:
        stopping["MaxWaitTimeInSeconds"] = sm_cfg["max_wait_min"] * 60

    sess.client("sagemaker").create_training_job(
        TrainingJobName=job,
        AlgorithmSpecification={"TrainingImage": sm_cfg["image_uri"], "TrainingInputMode": "File"},
        RoleArn=role,
        HyperParameters={k: json.dumps(v) for k, v in hp.items()},   # the SDK's encoding
        InputDataConfig=[{"ChannelName": "data", "DataSource": {"S3DataSource": {
            "S3DataType": "S3Prefix", "S3Uri": data_uri, "S3DataDistributionType": "FullyReplicated"}}}],
        OutputDataConfig={"S3OutputPath": f"s3://{bucket}/jobs/"},
        ResourceConfig={"InstanceType": sm_cfg["instance_type"], "InstanceCount": 1, "VolumeSizeInGB": 20},
        StoppingCondition=stopping,
        EnableManagedSpotTraining=sm_cfg["spot"],
        Environment={"PETCNN_PROCESSED": "/opt/ml/input/data/data", "PETCNN_MODELS": "/opt/ml/model"},
        Tags=TAGS,
    )
    print(f"started {job}; logs: CloudWatch /aws/sagemaker/TrainingJobs")
    if wait:
        status(job, wait=True)
    return job


def status(job: str, wait: bool = False) -> dict:
    sm = _session().client("sagemaker")
    while True:
        d = sm.describe_training_job(TrainingJobName=job)
        print(f"{job}: {d['TrainingJobStatus']} / {d.get('SecondaryStatus')}")
        if not wait or d["TrainingJobStatus"] in ("Completed", "Failed", "Stopped"):
            break
        time.sleep(60)
    if d["TrainingJobStatus"] == "Completed":
        billable = d.get("BillableTimeInSeconds", 0)
        rate = config.load()["prices"][d["ResourceConfig"]["InstanceType"]]
        print(f"  billable {billable}s (training {d.get('TrainingTimeInSeconds')}s) "
              f"~ ${billable / 3600 * rate:.3f} at on-demand rate; spot savings applied on the bill")
    elif d["TrainingJobStatus"] == "Failed":
        print(f"  {d.get('FailureReason')}")
    return d


def fetch(job: str, name: str) -> None:
    """Download the job's model.tar.gz into models/<name>/ (safetensors + JSON only)."""
    sess = _session()
    d = sess.client("sagemaker").describe_training_job(TrainingJobName=job)
    uri = d["ModelArtifacts"]["S3ModelArtifacts"]
    bucket, key = uri.removeprefix("s3://").split("/", 1)
    body = sess.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
    out = config.MODELS / name
    out.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(body)) as t:
        members = [m for m in t.getmembers() if m.isfile() and m.name.split("/")[-1] in
                   ("model.safetensors", "meta.json", "history.json")]
        for m in members:
            m.name = m.name.split("/")[-1]
        t.extractall(out, members=members, filter="data")
    print(f"fetched {uri} -> {out}: {[m.name for m in members]}")


# --- serving: model.tar.gz for the serverless endpoint ----------------------------------------

def model_tar(name: str) -> bytes:
    """model.safetensors + meta.json at the root, handler and its two imports under code/."""
    d = config.MODELS / name
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        t.add(d / "model.safetensors", "model.safetensors")
        t.add(d / "meta.json", "meta.json")
        t.add(config.ROOT / "serve" / "inference.py", "code/inference.py")
        t.add(config.ROOT / "serve" / "code-requirements.txt", "code/requirements.txt")
        for mod in ("models.py", "preprocess.py"):
            t.add(config.ROOT / "src" / "petcnn" / mod, f"code/{mod}")
    return buf.getvalue()


def package(name: str) -> str:
    """Upload models/<name> as model.tar.gz under a content-hash key; print and return its S3 URI."""
    import hashlib
    sess = _session()
    bucket = _stack_outputs(sess)["Bucket"]
    body = model_tar(name)
    key = f"models/{name}/{hashlib.sha256(body).hexdigest()[:16]}/model.tar.gz"
    sess.client("s3").put_object(Bucket=bucket, Key=key, Body=body)
    uri = f"s3://{bucket}/{key}"
    print(uri)
    return uri


def invoke(paths: list[str]) -> None:
    """Call the deployed endpoint directly with IAM credentials (bypasses the public API)."""
    sess = _session()
    cf = sess.client("cloudformation")
    outs = cf.describe_stacks(StackName=config.load()["serve"]["stack"])["Stacks"][0]["Outputs"]
    endpoint = {o["OutputKey"]: o["OutputValue"] for o in outs}["EndpointName"]
    rt = sess.client("sagemaker-runtime")
    for path in paths:
        t0 = time.time()
        r = rt.invoke_endpoint(EndpointName=endpoint, ContentType="image/jpeg", Accept="application/json",
                               Body=open(path, "rb").read())
        print(f"{path}: {r['Body'].read().decode()} ({time.time() - t0:.1f}s)")
