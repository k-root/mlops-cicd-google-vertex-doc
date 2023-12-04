from kfp import dsl
from kfp.dsl import importer, OneOf
from google_cloud_pipeline_components.v1.custom_job import CustomTrainingJobOp
from google_cloud_pipeline_components.types import artifact_types
from google_cloud_pipeline_components.v1.model import ModelUploadOp
from google_cloud_pipeline_components.v1.batch_predict_job import ModelBatchPredictOp
from google_cloud_pipeline_components.v1.model_evaluation import ModelEvaluationRegressionOp
from google_cloud_pipeline_components.v1.vertex_notification_email import VertexNotificationEmailOp    
from google.cloud import aiplatform
from kfp import compiler 

import os
PROJECT_ID = "springml-notebook-testing"  # @param {type:"string"}
REGION = "us-central1"  # @param {type: "string"}
BUCKET_NAME = f"{PROJECT_ID}-mlops-artifacts"
BUCKET_URI = f"gs://{BUCKET_NAME}"  # @param {type:"string"}
PIPELINE_ROOT = "{}/pipeline_root/chicago-taxi-pipe".format(BUCKET_URI)
UUID = "mlops-scheduler-tutorial"
PIPELINE_NAME = "vertex-pipeline-scheduler-tutorial"
WORKING_DIR = f"{PIPELINE_ROOT}/{UUID}"
os.environ['AIP_MODEL_DIR'] = WORKING_DIR
EXPERIMENT_NAME = PIPELINE_NAME + "-experiment"

# define the train-deploy pipeline
@dsl.pipeline(name="custom-model-training-evaluation-pipeline")
def custom_model_training_evaluation_pipeline(
    project: str,
    location: str,
    training_job_display_name: str,
    worker_pool_specs: list,
    base_output_dir: str,
    prediction_container_uri: str,
    model_display_name: str,
    batch_prediction_job_display_name: str,
    target_field_name: str,
    test_data_gcs_uri: list,
    ground_truth_gcs_source: list,
    batch_predictions_gcs_prefix: str,
    batch_predictions_input_format: str="csv",
    batch_predictions_output_format: str="jsonl",
    ground_truth_format: str="csv",
    parent_model_resource_name: str=None,
    parent_model_artifact_uri: str=None,
    existing_model: bool=False,
):
    # Notification task
    notify_task = VertexNotificationEmailOp(
                    recipients= ["kaushik.koilada@egen.ai"]
                    )
    with dsl.ExitHandler(notify_task, name='MLOps Scheduled Training Pipeline'):
        # Train the model
        custom_job_task = CustomTrainingJobOp(
                                    project=project,
                                    display_name=training_job_display_name,
                                    worker_pool_specs=worker_pool_specs,
                                    base_output_directory=base_output_dir,
                                    location=location
                            )

        # Import the unmanaged model
        import_unmanaged_model_task = importer(
                                        artifact_uri=base_output_dir,
                                        artifact_class=artifact_types.UnmanagedContainerModel,
                                        metadata={
                                            "containerSpec": {
                                                "imageUri": prediction_container_uri,
                                            },
                                        },
                                    ).after(custom_job_task)

        with dsl.If(existing_model == True):
            # Import the parent model to upload as a version
            import_registry_model_task = importer(
                                        artifact_uri=parent_model_artifact_uri,
                                        artifact_class=artifact_types.VertexModel,
                                        metadata={
                                            "resourceName": parent_model_resource_name
                                        },
                                    ).after(import_unmanaged_model_task)
            # Upload the model as a version
            model_version_upload_op = ModelUploadOp(
                                    project=project,
                                    location=location,
                                    display_name=model_display_name,
                                    parent_model=import_registry_model_task.outputs["artifact"],
                                    unmanaged_container_model=import_unmanaged_model_task.outputs["artifact"],
                                )

        with dsl.Else():
            # Upload the model
            model_upload_op = ModelUploadOp(
                                    project=project,
                                    location=location,
                                    display_name=model_display_name,
                                    unmanaged_container_model=import_unmanaged_model_task.outputs["artifact"],
                                )
        # Get the model (or model version)
        model_resource = OneOf(model_version_upload_op.outputs["model"], model_upload_op.outputs["model"])
        
        # Batch prediction
        batch_predict_task = ModelBatchPredictOp(
                            project= project, 
                            job_display_name= batch_prediction_job_display_name, 
                            model= model_resource, 
                            location= location, 
                            instances_format= batch_predictions_input_format, 
                            predictions_format= batch_predictions_output_format, 
                            gcs_source_uris= test_data_gcs_uri,  
                            gcs_destination_output_uri_prefix= batch_predictions_gcs_prefix, 
                            machine_type= 'n1-standard-2'
                            )
        # Evaluation task
        evaluation_task = ModelEvaluationRegressionOp(
                            project= project, 
                            target_field_name= target_field_name, 
                            location= location, 
                            # model= model_resource,
                            predictions_format= batch_predictions_output_format, 
                            predictions_gcs_source= batch_predict_task.outputs["gcs_output_directory"],
                            ground_truth_format= ground_truth_format, 
                            ground_truth_gcs_source= ground_truth_gcs_source
                            )
        
    return

worker_pool_specs = [{
                        "machine_spec": {"machine_type": "e2-highmem-2"},
                        "replica_count": 1,
                        "python_package_spec":{
                                "executor_image_uri": "us-docker.pkg.dev/vertex-ai/training/sklearn-cpu.1-0:latest", 
                                "package_uris": [f"{BUCKET_URI}/trainer-0.1.tar.gz"],
                                "python_module": "trainer.task",
                                "args":["--training-dir",f"/gcs/{BUCKET_NAME}"]
                        },
}]

parameters = {
    "project": PROJECT_ID,
    "location": REGION,
    "training_job_display_name": "taxifare-prediction-training-job",
    "worker_pool_specs": worker_pool_specs,
    "base_output_dir": BUCKET_URI,
    "prediction_container_uri": "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest",
    "model_display_name": "taxifare-prediction-model",
    "batch_prediction_job_display_name": "taxifare-prediction-batch-job",
    "target_field_name": "fare",
    "test_data_gcs_uri": [f"{BUCKET_URI}/test_no_target.csv"],
    "ground_truth_gcs_source": [f"{BUCKET_URI}/test.csv"],
    "batch_predictions_gcs_prefix": f"{BUCKET_URI}/batch_predict_output",
    # "batch_predictions_input_format": "csv",
    # "batch_predictions_output_format": "jsonl",
    # "ground_truth_format": "csv",
    # "parent_model_resource_name": None,
    # "parent_model_artifact_uri": None,
    "existing_model": False
}


compiler.Compiler().compile(
    pipeline_func=custom_model_training_evaluation_pipeline,
    package_path="pipeline.yaml",
)

# aiplatform.init(project= "springml-notebook-testing", 
#                  location= "us-central1")

aiplatform.init(
    project=PROJECT_ID, 
    staging_bucket=BUCKET_URI, 
    location=REGION,
    experiment=EXPERIMENT_NAME)

aiplatform.autolog()

job = aiplatform.PipelineJob(
    display_name="scheduled_custom_regression_evaluation",
    template_path="pipeline.yaml",
    parameter_values=parameters,
    pipeline_root=BUCKET_URI,
    enable_caching=True
)
job.submit(experiment=EXPERIMENT_NAME)