import kfp
from google.cloud import aiplatform
from google_cloud_pipeline_components.types import artifact_types
from kfp import compiler,dsl
from kfp.components import load_component_from_file
from kfp.dsl import Input, Metrics, component, Output, Model, Dataset

PROJECT_ID = "springml-notebook-testing"  # @param {type:"string"}
REGION = "us-central1"  # @param {type: "string"}
BUCKET_URI = f"gs://{PROJECT_ID}-mlops-artifacts"  # @param {type:"string"}
PIPELINE_ROOT = "{}/pipeline_root/chicago-taxi-pipe".format(BUCKET_URI)
UUID = "mlops-scheduler-tutorial"
PIPELINE_NAME = "vertex-pipeline-ct-tutorial-deploy"
WORKING_DIR = f"{PIPELINE_ROOT}/{UUID}"
EXPERIMENT_NAME = PIPELINE_NAME + "-experiment"



@kfp.dsl.pipeline(name= UUID)
def pipeline(
    metric: str,
    threshold: float,
    project: str = PROJECT_ID
):
    from google_cloud_pipeline_components.v1.vertex_notification_email import VertexNotificationEmailOp
    from google_cloud_pipeline_components.v1.endpoint import (EndpointCreateOp,
                                                              ModelDeployOp)

    notify_email_task = VertexNotificationEmailOp(recipients=["kaushik.koilada@springml.com"])
    with dsl.ExitHandler(notify_email_task, name='MLOps Continuous Training Pipeline'):
        process_data_loaded_component = load_component_from_file('process_data.yaml')
        train_model_loaded_component = load_component_from_file('train_model.yaml')
        run_model_loaded_component = load_component_from_file('run_model.yaml')
        evaluate_model_loaded_component = load_component_from_file('evaluate_model.yaml')

        data_prep = process_data_loaded_component(project_id=project)
        training_step = train_model_loaded_component(model_directory=WORKING_DIR,data=data_prep.output)
        model_predictions = run_model_loaded_component(final_model=training_step.outputs["model_artifact"],data=data_prep.output)
        model_evaluation = evaluate_model_loaded_component(metric_name=metric, metric_threshold=threshold, y=model_predictions.output)
        
        # with dsl.If(model_evaluation.outputs['threshold_flag'] == "true", name="Evaluation Threshold Pass"):            
        upload_model_to_vertex_component = load_component_from_file('upload_model_to_vertex.yaml')
        upload_model = upload_model_to_vertex_component(project = project, final_model=training_step.outputs["model_artifact"],display_name="mlops-vertex-ct-model")
            
#             endpoint_create_op = EndpointCreateOp(
#                 project=project,
#                 display_name="mlops-ct-automated-endpoint",
#             )

#             ModelDeployOp(
#                 endpoint=endpoint_create_op.outputs["endpoint"],
#                 model=aiplatform.Model(model_name=upload_model.outputs['model_name']),
#                 deployed_model_display_name="chicago-taxi-fare-pred-model",
#                 dedicated_resources_machine_type="n1-standard-16",
#                 dedicated_resources_min_replica_count=1,
#                 dedicated_resources_max_replica_count=1,
#             )

        print(model_evaluation)

def upload_to_gcs(gcs_file_path: str, local_path: str):
    storage_path = os.path.join(gcs_file_path, local_path)
    blob = storage.blob.Blob.from_string(storage_path, client=storage.Client())
    blob.upload_from_filename(local_path)
    print("Upload of {} to GCS location {} complete".format(local_path,storage_path))
    

if __name__=="__main__": 
    
    compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path="./{}.yaml".format(PIPELINE_NAME),
    )
    upload_to_gcs(WORKING_DIR
                  ,"./{}.yaml".format(PIPELINE_NAME))
    DISPLAY_NAME = PIPELINE_NAME

    job = aiplatform.PipelineJob(
        display_name=PIPELINE_NAME,
        template_path="{}.json".format(PIPELINE_NAME),
        pipeline_root=PIPELINE_ROOT,
        parameter_values={"metric": "r2_score", "threshold": 0.7},
        enable_caching=True

    )
