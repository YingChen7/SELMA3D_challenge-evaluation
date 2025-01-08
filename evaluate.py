"""
The following is a simple example evaluation method.

It is meant to run within a container.

To run it locally, you can call the following bash script:

  ./test_run.sh

This will start the evaluation, reads from ./test/input and outputs to ./test/output

To save the container and prep it for upload to Grand-Challenge.org you can call:

  ./save.sh

Any container that shows the same behavior will do, this is purely an example of how one COULD do it.

Happy programming!
"""
import json
from glob import glob
import SimpleITK
import random
from multiprocessing import Pool
from statistics import mean
from pathlib import Path
from pprint import pformat, pprint
import numpy as np
from metrics import volumetric_dice, centerline_dice, betti_number

INPUT_DIRECTORY = Path("./input")
OUTPUT_DIRECTORY = Path("./output")
GROUND_TRUTH_DIRECTORY = Path("./ground_truth")


def main():
    print_inputs() # what is in the test input

    metrics = {}
    predictions = read_predictions()
    # We now process each algorithm job for this submission
    # Note that the jobs are not in any order!
    # We work that out from predictions.json

    # Start a number of process workers, using multiprocessing
    # The optimal number of workers ultimately depends on how many
    # resources each process() would call upon
    with Pool(processes=4) as pool:
        metrics["results"] = pool.map(process, predictions)

    # Now generate an overall score(s) for this submission
    cfos_dice_scores, cfos_b0_errors = [], []

    micro_dice_scores, micro_b0_errors, micro_b1_errors, micro_cldice_scores = [], [], [], []
    for result in metrics["results"]:
        if len(result) == 2:
            cfos_dice_scores.append(result["dice"])
            cfos_b0_errors.append(result["betti_matching_error_0"])
        else:
            micro_dice_scores.append(result["dice"])
            micro_b0_errors.append(result["betti_matching_error_0"])
            micro_b1_errors.append(result["betti_matching_error_1"])
            micro_cldice_scores.append(result["centerline_dice"])

    metrics["aggregates"] = {
        "cfos_dice": mean(cfos_dice_scores),
        "cfos_betti_error_0": mean(cfos_b0_errors),
        "micro_dice": mean(micro_dice_scores),
        "micro_betti_error_0": mean(micro_b0_errors),
        "micro_betti_error_1": mean(micro_b1_errors),
        "micro_cldice": mean(micro_cldice_scores)
    }

    # Make sure to save the metrics
    write_metrics(metrics=metrics)

    return 0


def process(job):
    # Processes a single algorithm job, looking at the outputs
    report = "Processing:\n"
    report += pformat(job)
    report += "\n"


    # Firstly, find the location of the results
    biological_brain_structure_location = get_file_location(
            job_pk=job["pk"],
            values=job["outputs"],
            slug="biological-brain-structure",
        )

    # Secondly, read the results
    biological_brain_structure = load_image_file(
        location=biological_brain_structure_location,
    )
    print(np.unique(biological_brain_structure))

    # Thirdly, retrieve the input image name to match it with an image in your ground truth
    brain_microscopy_image_name = get_image_name(
            values=job["inputs"],
            slug="3d-brain-microscopy",
    )
    print('brain_microscopy_image_name:', brain_microscopy_image_name)

    # Fourthly, your load your ground truth
    # Include it in your evaluation container by placing it in ground_truth/

    gt = SimpleITK.ReadImage(str(GROUND_TRUTH_DIRECTORY / brain_microscopy_image_name))
    gt_arr = SimpleITK.GetArrayFromImage(gt)
    print(np.unique(gt_arr))
    if 'cfos' in str(brain_microscopy_image_name):
        dice_value = volumetric_dice.volume_dice(gt_arr, biological_brain_structure)
        betti_numbers = betti_number.betti_number_error(gt_arr, biological_brain_structure)
        print('here')
        return {
            "dice": dice_value,
            "betti_matching_error_0":betti_numbers[0]
        }
        
    elif 'microglia' in str(brain_microscopy_image_name):
        dice_value = volumetric_dice.volume_dice(gt_arr, biological_brain_structure)
        betti_numbers = betti_number.betti_number_error(gt_arr, biological_brain_structure)
        cldice_value = centerline_dice.centerline_dice(gt_arr, biological_brain_structure)
        return {
            "dice": dice_value,
            "betti_matching_error_0":betti_numbers[0],
            "betti_matching_error_1":betti_numbers[1],
            "centerline_dice": cldice_value
        }
    else:
       return {
            "my_metric": 0
        }
    ''''
    with open(GROUND_TRUTH_DIRECTORY / "some_resource.txt", "r") as f:
        report += f.read()
        # compare the results to your ground truth and compute some metrics

    print(report)


    # Finally, calculate by comparing the ground truth to the actual results
    return {
        "my_metric": random.choice([1, 0]),
    }
    '''

def print_inputs():
    # Just for convenience, in the logs you can then see what files you have to work with
    input_files = [str(x) for x in Path(INPUT_DIRECTORY).rglob("*") if x.is_file()]

    print("Input Files:")
    pprint(input_files)
    print("")


def read_predictions():
    # The prediction file tells us the location of the users' predictions
    with open(INPUT_DIRECTORY / "predictions.json") as f:
        return json.loads(f.read())


def get_image_name(*, values, slug):
    # This tells us the user-provided name of the input or output image
    for value in values:
        if value["interface"]["slug"] == slug:
            return value["image"]["name"]

    raise RuntimeError(f"Image with interface {slug} not found!")


def get_interface_relative_path(*, values, slug):
    # Gets the location of the interface relative to the input or output
    for value in values:
        if value["interface"]["slug"] == slug:
            return value["interface"]["relative_path"]

    raise RuntimeError(f"Value with interface {slug} not found!")


def get_file_location(*, job_pk, values, slug):
    # Where a job's output file will be located in the evaluation container
    relative_path = get_interface_relative_path(values=values, slug=slug)
    return INPUT_DIRECTORY / job_pk / "output" / relative_path


def load_image_file(*, location):
    # Use SimpleITK to read a file
    input_files = glob(str(location / "*.nii.gz")) + glob(str(location / "*.mha"))
    result = SimpleITK.ReadImage(input_files[0])
    # Convert it to a Numpy array
    return SimpleITK.GetArrayFromImage(result)


def write_metrics(*, metrics):
    # Write a json document used for ranking results on the leaderboard
    
    with open(OUTPUT_DIRECTORY / "metrics.json", "w") as f:
        f.write(json.dumps(metrics, indent=4))


if __name__ == "__main__":
    raise SystemExit(main())
