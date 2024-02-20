import torch
import numpy as np
import csv


import pytesseract


from PIL import Image
from torchvision import transforms
from transformers import TableTransformerForObjectDetection, AutoModelForObjectDetection
from tqdm.auto import tqdm


from skimage.filters import threshold_otsu, rank
from skimage.morphology import disk
import skimage.io as io
from skimage.color import rgb2gray


class TableRecognition:
    """
    This is a class for detecting and recognizing tables from png images.
    """

    def __init__(self) -> None:
        # Loading the models
        self.model = AutoModelForObjectDetection.from_pretrained(
            "microsoft/table-transformer-detection", revision="no_timm"
        )
        # This code based on next sources:
        # https://huggingface.co/docs/transformers/main/en/model_doc/table-transformer
        # https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Table%20Transformer/Inference_with_Table_Transformer_(TATR)_for_parsing_tables.ipynb
        self.structure_model = TableTransformerForObjectDetection.from_pretrained(
            "microsoft/table-structure-recognition-v1.1-all"
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.structure_model.to(self.device)

        self.detection_transform = transforms.Compose(
            [
                self.MaxResize(800),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.structure_transform = transforms.Compose(
            [
                self.MaxResize(1000),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    # def local_otsu(file_path = ".\pic\pic.png"):
    #     img = io.imread(file_path)[:, :, :-1]
    #     img = rgb2gray(img)
    #     radius = 1
    #     footprint = disk(radius)
    #     local_otsu = rank.otsu(img, footprint)
    #     return img >= local_otsu

    def otsu(file_path=".\pic\pic.png"):
        """
        Method for preprosess the image.
        """
        image = io.imread(file_path)[:, :, :-1]
        image = rgb2gray(image)
        thresh = threshold_otsu(image)
        binary = image > thresh
        return binary

    class MaxResize(object):
        """
        Class for preparing img for model.
        """

        def __init__(self, max_size=800):
            self.max_size = max_size

        def __call__(self, image):
            width, height = image.size
            current_max_size = max(width, height)
            scale = self.max_size / current_max_size
            resized_image = image.resize(
                (int(round(scale * width)), int(round(scale * height)))
            )

            return resized_image

    def box_cxcywh_to_xyxy(self, x):
        """
        Method for change coordinates of the bounding boxes from
        `(x_center, y_center, width, hight)` -->
        `(x_l_upper_conner, y_l_upper_conner, x_r_down_conner, y_r_down_conner)`
        """
        x_c, y_c, w, h = x.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)

    def rescale_bboxes(self, out_bbox, size):
        """
        Method for rescalling bounding boxes to the original size.
        """
        img_w, img_h = size
        b = self.box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return b

    def outputs_to_objects(self, outputs, img_size, id2label):
        """
        Transforming output into clear form.
        """
        m = outputs.logits.softmax(-1).max(-1)
        pred_labels = list(m.indices.detach().cpu().numpy())[0]
        pred_scores = list(m.values.detach().cpu().numpy())[0]
        pred_bboxes = outputs["pred_boxes"].detach().cpu()[0]
        pred_bboxes = [
            elem.tolist() for elem in self.rescale_bboxes(pred_bboxes, img_size)
        ]

        objects = []
        for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
            class_label = id2label[int(label)]
            if not class_label == "no object":
                objects.append(
                    {
                        "label": class_label,
                        "score": float(score),
                        "bbox": [float(elem) for elem in bbox],
                    }
                )

        return objects

    def table_detection(self, image):
        """
        Applying table detection model.
        """
        pixel_values = self.detection_transform(image).unsqueeze(0)
        pixel_values = pixel_values.to(self.device)
        with torch.no_grad():
            outputs = self.model(pixel_values)
        # update id2label to include "no object"
        id2label = self.model.config.id2label
        id2label[len(self.model.config.id2label)] = "no object"
        objects = self.outputs_to_objects(outputs, image.size, id2label)
        return objects

    def detection(self, file_path, preprocessing=None):
        if preprocessing is not None:
            image = self.otsu(file_path)
            image = Image.fromarray(image).convert("RGB")
        else:
            image = Image.open(file_path).convert("RGB")
        table = self.table_detection(image)

        cropped_table = self.objects_to_crops(image, table)

        pixel_values = self.structure_transform(cropped_table).unsqueeze(0)
        pixel_values = pixel_values.to(self.device)
        with torch.no_grad():
            outputs = self.structure_model(pixel_values)
        structure_id2label = self.structure_model.config.id2label
        structure_id2label[len(structure_id2label)] = "no object"

        cells = self.outputs_to_objects(outputs, cropped_table.size, structure_id2label)
        cell_coordinates = self.get_cell_coordinates_by_row(cells)
        data = self.apply_ocr(cell_coordinates, cropped_table)
        with open(file_path[:-3] + "csv", "w", encoding="utf-8") as result_file:
            wr = csv.writer(
                result_file,
                dialect="excel",
            )
            for row, row_text in data.items():
                wr.writerow([el.strip().replace("\n", " ") for el in row_text])

    # Crop the detected table from the image
    def objects_to_crops(
        self,
        img,
        objects,
        padding=10,
        tokens=[],
        class_thresholds={"table": 0.5, "table rotated": 0.5, "no object": 10},
    ):
        """
        Process the bounding boxes produced by the table detection model into
        cropped table images and cropped tokens.
        """

        table_crops = []
        for obj in objects:
            if obj["score"] < class_thresholds[obj["label"]]:
                continue

            cropped_table = {}

            bbox = obj["bbox"]
            bbox = [
                bbox[0] - padding,
                bbox[1] - padding,
                bbox[2] + padding,
                bbox[3] + padding,
            ]

            cropped_img = img.crop(bbox)

            table_tokens = [
                token for token in tokens if iob(token["bbox"], bbox) >= 0.5
            ]
            for token in table_tokens:
                token["bbox"] = [
                    token["bbox"][0] - bbox[0],
                    token["bbox"][1] - bbox[1],
                    token["bbox"][2] - bbox[0],
                    token["bbox"][3] - bbox[1],
                ]

            # If table is predicted to be rotated, rotate cropped image and tokens/words:
            if obj["label"] == "table rotated":
                cropped_img = cropped_img.rotate(270, expand=True)
                for token in table_tokens:
                    bbox = token["bbox"]
                    bbox = [
                        cropped_img.size[0] - bbox[3] - 1,
                        bbox[0],
                        cropped_img.size[0] - bbox[1] - 1,
                        bbox[2],
                    ]
                    token["bbox"] = bbox

            cropped_table["image"] = cropped_img
            cropped_table["tokens"] = table_tokens

            table_crops.append(cropped_table)

        return table_crops[0]["image"]

    def get_cell_coordinates_by_row(self, table_data):

        # Extract rows and columns
        rows = [entry for entry in table_data if entry["label"] == "table row"]
        columns = [entry for entry in table_data if entry["label"] == "table column"]

        # Sort rows and columns by their Y and X coordinates, respectively
        rows.sort(key=lambda x: x["bbox"][1])
        columns.sort(key=lambda x: x["bbox"][0])

        # Function to find cell coordinates
        def find_cell_coordinates(row, column):
            cell_bbox = [
                column["bbox"][0],
                row["bbox"][1],
                column["bbox"][2],
                row["bbox"][3],
            ]
            return cell_bbox

        # Generate cell coordinates and count cells in each row
        cell_coordinates = []

        for row in rows:
            row_cells = []
            for column in columns:
                cell_bbox = find_cell_coordinates(row, column)
                row_cells.append({"column": column["bbox"], "cell": cell_bbox})

            # Sort cells in the row by X coordinate
            row_cells.sort(key=lambda x: x["column"][0])

            # Append row information to cell_coordinates
            cell_coordinates.append(
                {"row": row["bbox"], "cells": row_cells, "cell_count": len(row_cells)}
            )

        # Sort rows from top to bottom
        cell_coordinates.sort(key=lambda x: x["row"][1])

        return cell_coordinates

    def apply_ocr(self, cell_coordinates, cropped_table):
        """
        Applying Pytesseract.
        """
        # let's OCR row by row
        data = dict()
        max_num_columns = 0
        for idx, row in enumerate(tqdm(cell_coordinates)):
            row_text = []
            for cell in row["cells"]:
                # crop cell out of image
                cell_image = np.array(cropped_table.crop(cell["cell"]))
                # apply OCR
                result = pytesseract.image_to_string(
                    image=np.array(cell_image), lang="rus"
                )
                if len(result) > 0:
                    # print([x[1] for x in list(result)])
                    text = "".join([x for x in result])
                    row_text.append(text)

            if len(row_text) > max_num_columns:
                max_num_columns = len(row_text)

            data[idx] = row_text

        # print("Max number of columns:", max_num_columns)

        # pad rows which don't have max_num_columns elements
        # to make sure all rows have the same number of columns
        for row, row_data in data.copy().items():
            if len(row_data) != max_num_columns:
                row_data = row_data + [
                    "" for _ in range(max_num_columns - len(row_data))
                ]
            data[row] = row_data
        return data
