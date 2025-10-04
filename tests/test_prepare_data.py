"""
Unit tests for data preparation pipeline
Tests bbox conversion, frame extraction, and YOLO format generation
"""

import pytest
import json
import cv2
import numpy as np
import tempfile
import os
from pathlib import Path
import sys

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from prepare_data import convert_bbox_to_yolo


class TestBboxConversion:
    """Test bounding box format conversions"""

    def test_convert_bbox_center_position(self):
        """Test conversion of bbox at center of image"""
        # Image 1920x1080, bbox at center: 100x100
        bbox = [910, 490, 100, 100]  # x, y, w, h (top-left format)
        img_width, img_height = 1920, 1080

        result = convert_bbox_to_yolo(bbox, img_width, img_height)

        # Expected: center at (960, 540), normalized
        expected_x_center = 960 / 1920  # 0.5
        expected_y_center = 540 / 1080  # 0.5
        expected_w = 100 / 1920  # ~0.052
        expected_h = 100 / 1080  # ~0.093

        assert abs(result[0] - expected_x_center) < 1e-5
        assert abs(result[1] - expected_y_center) < 1e-5
        assert abs(result[2] - expected_w) < 1e-5
        assert abs(result[3] - expected_h) < 1e-5

    def test_convert_bbox_top_left_corner(self):
        """Test conversion of bbox at top-left corner"""
        bbox = [0, 0, 50, 50]
        img_width, img_height = 1920, 1080

        result = convert_bbox_to_yolo(bbox, img_width, img_height)

        # Center should be at (25, 25) normalized
        expected_x_center = 25 / 1920
        expected_y_center = 25 / 1080

        assert abs(result[0] - expected_x_center) < 1e-5
        assert abs(result[1] - expected_y_center) < 1e-5

    def test_convert_bbox_bottom_right_corner(self):
        """Test conversion of bbox at bottom-right corner"""
        bbox = [1870, 1030, 50, 50]  # Bottom-right corner
        img_width, img_height = 1920, 1080

        result = convert_bbox_to_yolo(bbox, img_width, img_height)

        # Center should be at (1895, 1055) normalized
        expected_x_center = 1895 / 1920
        expected_y_center = 1055 / 1080

        assert abs(result[0] - expected_x_center) < 1e-5
        assert abs(result[1] - expected_y_center) < 1e-5

    def test_convert_bbox_values_in_range(self):
        """Test that all converted values are in [0, 1] range"""
        test_cases = [
            ([100, 100, 200, 150], 1920, 1080),
            ([0, 0, 100, 100], 1920, 1080),
            ([1820, 980, 100, 100], 1920, 1080),
            ([960, 540, 50, 50], 1920, 1080),
        ]

        for bbox, w, h in test_cases:
            result = convert_bbox_to_yolo(bbox, w, h)
            assert 0 <= result[0] <= 1, f"x_center out of range: {result[0]}"
            assert 0 <= result[1] <= 1, f"y_center out of range: {result[1]}"
            assert 0 <= result[2] <= 1, f"width out of range: {result[2]}"
            assert 0 <= result[3] <= 1, f"height out of range: {result[3]}"

    def test_convert_bbox_small_object(self):
        """Test conversion of very small bounding box"""
        bbox = [500, 300, 10, 10]  # Small 10x10 box
        img_width, img_height = 1920, 1080

        result = convert_bbox_to_yolo(bbox, img_width, img_height)

        # Should still have valid normalized values
        assert result[2] > 0  # width should be positive
        assert result[3] > 0  # height should be positive
        assert result[2] < 0.01  # but small relative to image

    def test_convert_bbox_large_object(self):
        """Test conversion of large bounding box"""
        bbox = [100, 100, 1720, 880]  # Large box covering most of image
        img_width, img_height = 1920, 1080

        result = convert_bbox_to_yolo(bbox, img_width, img_height)

        # Should have large normalized dimensions
        assert result[2] > 0.8  # width should be large
        assert result[3] > 0.8  # height should be large
        assert result[0] == 0.5  # centered horizontally
        assert abs(result[1] - 0.5) < 0.05  # centered vertically

    def test_convert_bbox_realistic_drone(self):
        """Test with realistic drone detection bbox from dataset"""
        # Real example from Anti-UAV dataset
        bbox = [786, 566, 135, 77]  # Typical small drone
        img_width, img_height = 1920, 1080

        result = convert_bbox_to_yolo(bbox, img_width, img_height)

        # Verify it's a small object near center
        assert 0.4 < result[0] < 0.6  # Near horizontal center
        assert 0.4 < result[1] < 0.6  # Near vertical center
        assert result[2] < 0.1  # Small relative width
        assert result[3] < 0.1  # Small relative height


class TestYOLOFormatValidation:
    """Test YOLO format output validation"""

    def test_yolo_format_structure(self):
        """Test that YOLO format has correct structure"""
        bbox = [786, 566, 135, 77]
        img_width, img_height = 1920, 1080

        result = convert_bbox_to_yolo(bbox, img_width, img_height)

        # Should return list of 4 floats
        assert isinstance(result, list)
        assert len(result) == 4
        assert all(isinstance(x, float) for x in result)

    def test_yolo_label_file_format(self):
        """Test that label file format is correct"""
        # Simulate label file content
        bbox = [786, 566, 135, 77]
        img_width, img_height = 1920, 1080
        yolo_bbox = convert_bbox_to_yolo(bbox, img_width, img_height)

        # Format as YOLO label line
        label_line = f"0 {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}"

        # Parse it back
        parts = label_line.split()
        assert len(parts) == 5
        assert parts[0] == "0"  # class_id
        assert all(0 <= float(p) <= 1 for p in parts[1:])  # normalized values


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_bbox_at_image_boundary(self):
        """Test bbox exactly at image boundaries"""
        # Bbox at right edge
        bbox = [1820, 540, 100, 100]
        img_width, img_height = 1920, 1080

        result = convert_bbox_to_yolo(bbox, img_width, img_height)

        # Should handle gracefully even if extends beyond image
        assert isinstance(result, list)
        assert len(result) == 4

    def test_different_aspect_ratios(self):
        """Test with different image aspect ratios"""
        test_cases = [
            ([100, 100, 50, 50], 1920, 1080),  # 16:9
            ([100, 100, 50, 50], 640, 480),    # 4:3
            ([100, 100, 50, 50], 1024, 1024),  # 1:1 square
            ([100, 100, 50, 50], 3840, 2160),  # 4K
        ]

        for bbox, w, h in test_cases:
            result = convert_bbox_to_yolo(bbox, w, h)
            assert 0 <= result[0] <= 1
            assert 0 <= result[1] <= 1
            assert 0 <= result[2] <= 1
            assert 0 <= result[3] <= 1


class TestIntegration:
    """Integration tests for full pipeline"""

    def test_mock_frame_annotation_pipeline(self):
        """Test complete frame + annotation processing pipeline"""
        # Create mock data
        mock_annotations = {
            "exist": [1, 1, 0, 1],
            "gt_rect": [
                [100, 100, 50, 50],
                [200, 200, 60, 60],
                [0, 0, 0, 0],  # Frame where drone doesn't exist
                [300, 300, 70, 70]
            ]
        }

        img_width, img_height = 1920, 1080

        # Process each frame
        results = []
        for frame_idx, (exists, bbox) in enumerate(zip(mock_annotations["exist"], mock_annotations["gt_rect"])):
            if exists == 1:
                yolo_bbox = convert_bbox_to_yolo(bbox, img_width, img_height)
                results.append({
                    "frame": frame_idx,
                    "bbox": yolo_bbox
                })

        # Verify results
        assert len(results) == 3  # 3 frames with detections
        assert results[0]["frame"] == 0
        assert results[1]["frame"] == 1
        assert results[2]["frame"] == 3

        # Verify bbox validity
        for r in results:
            assert all(0 <= x <= 1 for x in r["bbox"])


class TestPerformance:
    """Performance and regression tests"""

    def test_bbox_conversion_precision(self):
        """Test that conversion maintains sufficient precision"""
        bbox = [786.5, 566.3, 135.7, 77.2]
        img_width, img_height = 1920, 1080

        result = convert_bbox_to_yolo(bbox, img_width, img_height)

        # Convert back to verify precision
        x_center_px = result[0] * img_width
        y_center_px = result[1] * img_height
        w_px = result[2] * img_width
        h_px = result[3] * img_height

        # Calculate original center
        orig_x_center = bbox[0] + bbox[2] / 2
        orig_y_center = bbox[1] + bbox[3] / 2

        # Should be accurate within 0.1 pixel
        assert abs(x_center_px - orig_x_center) < 0.1
        assert abs(y_center_px - orig_y_center) < 0.1
        assert abs(w_px - bbox[2]) < 0.1
        assert abs(h_px - bbox[3]) < 0.1

    def test_batch_processing(self):
        """Test processing multiple bboxes efficiently"""
        img_width, img_height = 1920, 1080
        # Generate bboxes that stay within image bounds
        bboxes = [
            [100 + (i % 15)*100, 100 + (i % 8)*100, 50, 50]
            for i in range(100)
        ]

        results = [convert_bbox_to_yolo(bbox, img_width, img_height) for bbox in bboxes]

        assert len(results) == 100
        assert all(len(r) == 4 for r in results)
        # Verify all values in valid range
        for i, r in enumerate(results):
            for j, x in enumerate(r):
                assert 0 <= x <= 1, f"Value out of range at result {i}, position {j}: {x}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
