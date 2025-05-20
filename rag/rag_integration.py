"""
RAG Integration Module

This module implements a RAG architecture using a local language model for 3D surface
processing and pattern generation. It generates grinding trajectory patterns
for robot arm movement based on 3D object boundaries, without requiring a Milvus server.

The system provides coordinates in the format (x, y, z, Rx, Ry, Rz) where
Rx, Ry, Rz are Euler angles for end-effector orientation.
"""

import os
import json
import numpy as np
import pickle
import re
import time
import torch
import traceback
import uuid
from scipy.spatial.distance import cosine
from typing import List, Dict, Tuple, Any
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


from misc.logger import Logger, Timer
from config.constants import (
    MODELS_DIR,
    DATA_DIR,
    RAG_PATTERNS_DIR,
    RAG_EMBEDDINGS_DIR,
    RAG_MODEL_NAME,
    RAG_TOP_K,
)


logger = Logger.get_logger("rag_integration", console_output=True, file_output=True)


class SurfacePatternRetriever:
    """
    Class for retrieving surface grinding patterns based on 3D object boundaries
    using RAG (Retrieval Augmented Generation) with a local model and file-based storage.
    """

    def __init__(self, model_name: str = RAG_MODEL_NAME, data_dir: str = DATA_DIR):
        """
        Initialize the SurfacePatternRetriever with the local model
        and file-based storage system.

        Args:
            model_name: Name of the model to use from HuggingFace
            data_dir: Directory to store pattern data and embeddings
        """
        logger.info("Initializing SurfacePatternRetriever")
        logger.info(f"Model name: {model_name}")
        logger.info(f"Data directory: {data_dir}")

        self.model_name = model_name
        self.data_dir = data_dir
        self.patterns_dir = RAG_PATTERNS_DIR
        self.embeddings_dir = RAG_EMBEDDINGS_DIR

        with Timer("Storage initialization", logger):
            try:
                self._init_storage()
                logger.info("Storage initialization successful")
            except Exception as e:
                logger.error(f"Failed to initialize storage: {str(e)}")
                logger.error(traceback.format_exc())
                raise

        with Timer("Model loading", logger):
            try:
                self._load_model()
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load model: {str(e)}")
                logger.error(traceback.format_exc())
                raise

    def _init_storage(self):
        """Initialize file-based storage directories"""
        logger.info("Initializing file-based storage")

        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.patterns_dir, exist_ok=True)
        os.makedirs(self.embeddings_dir, exist_ok=True)

        logger.info(f"Created patterns directory at {self.patterns_dir}")
        logger.info(f"Created embeddings directory at {self.embeddings_dir}")

        self.index_file = os.path.join(self.data_dir, "pattern_index.json")
        if not os.path.exists(self.index_file):
            logger.info(f"Creating new pattern index at {self.index_file}")
            with open(self.index_file, "w") as f:
                json.dump([], f)
        else:
            logger.info(f"Using existing pattern index at {self.index_file}")

        self._load_index()

    def _load_index(self):
        """Load pattern index from file"""
        try:
            with open(self.index_file, "r") as f:
                self.pattern_index = json.load(f)
            logger.info(f"Loaded {len(self.pattern_index)} patterns from index")
        except Exception as e:
            logger.error(f"Error loading pattern index: {str(e)}")
            logger.error(traceback.format_exc())
            # Create empty index if loading fails
            self.pattern_index = []

    def _save_index(self):
        """Save pattern index to file"""
        try:
            with open(self.index_file, "w") as f:
                json.dump(self.pattern_index, f, indent=2)
            logger.info(f"Saved {len(self.pattern_index)} patterns to index")
        except Exception as e:
            logger.error(f"Error saving pattern index: {str(e)}")
            logger.error(traceback.format_exc())

    def _load_model(self):
        """Load the local model from HuggingFace for generating embeddings and patterns"""
        logger.info(f"Loading model {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, cache_dir=MODELS_DIR
        )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name, cache_dir=MODELS_DIR
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        logger.info(f"Model loaded on device: {self.device}")

    def _get_embedding(
        self, boundary_points: List[Tuple[float, float, float]]
    ) -> np.ndarray:
        """
        Generate embedding for boundary points using the model

        Args:
            boundary_points: List of (x, y, z) coordinates that define the boundary

        Returns:
            Vector embedding as numpy array
        """
        logger.info(f"Generating embedding for {len(boundary_points)} boundary points")

        boundary_str = self._format_points_for_model(boundary_points)

        inputs = self.tokenizer(
            boundary_str, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.get_encoder()(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

        logger.info(f"Embedding generated with shape: {embedding.shape}")
        return embedding[0]

    def _format_points_for_model(self, points: List[Tuple[float, float, float]]) -> str:
        """
        Format 3D points into a string representation for the model

        Args:
            points: List of 3D coordinates

        Returns:
            Formatted string representation
        """
        points_str = "3D_POINTS: " + " ".join(
            [f"({x:.3f},{y:.3f},{z:.3f})" for x, y, z in points]
        )
        logger.debug(f"Formatted points string: {points_str[:100]}...")
        return points_str

    def _generate_pattern(
        self,
        boundary_points: List[Tuple[float, float, float]],
        similar_patterns: List[Dict],
    ) -> List[Tuple[float, float, float, float, float, float]]:
        """
        Generate grinding pattern based on boundary points and similar patterns

        Args:
            boundary_points: List of boundary point coordinates
            similar_patterns: List of similar pattern dictionaries from storage

        Returns:
            List of pattern points (x, y, z, Rx, Ry, Rz)
        """
        logger.info("Generating pattern using model")

        similar_pattern_strs = []
        for pattern in similar_patterns:
            if "pattern_points" in pattern:
                pattern_str = self._points_to_str(pattern["pattern_points"])
                similar_pattern_strs.append(pattern_str)

        prompt = self._create_generation_prompt(boundary_points, similar_pattern_strs)

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                inputs.input_ids,
                max_length=2048,
                temperature=0.7,
                top_p=0.95,
                do_sample=True,
                num_return_sequences=1,
                repetition_penalty=1.2,
                length_penalty=1.0,
            )

        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        preview = (
            generated_text[:200] + "..."
            if len(generated_text) > 200
            else generated_text
        )
        logger.debug(f"Generated text preview: {preview}")

        pattern_points = self._parse_pattern_from_text(generated_text)

        if not pattern_points and len(boundary_points) > 0:
            logger.warning(
                "No pattern points parsed from first attempt, trying with more explicit prompt"
            )
            explicit_prompt = self._create_explicit_prompt(boundary_points)

            inputs = self.tokenizer(
                explicit_prompt, return_tensors="pt", truncation=True, max_length=1024
            ).to(self.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    inputs.input_ids,
                    max_length=2048,
                    temperature=0.5,  # Lower temperature for more focused
                    top_p=0.9,
                    do_sample=True,
                    num_return_sequences=1,
                    repetition_penalty=1.5,
                )

            explicit_text = self.tokenizer.decode(
                output_ids[0], skip_special_tokens=True
            )
            pattern_points = self._parse_pattern_from_text(explicit_text)

        logger.info(f"Generated pattern with {len(pattern_points)} points")
        return pattern_points

    def _create_explicit_prompt(
        self, boundary_points: List[Tuple[float, float, float]]
    ) -> str:
        """
        Create a more explicit prompt for pattern generation when the first attempt fails

        Args:
            boundary_points: List of boundary point coordinates

        Returns:
            More explicit prompt for the model
        """
        # Calculate centroid and bounds
        xs, ys, zs = zip(*boundary_points)
        center_x = sum(xs) / len(xs)
        center_y = sum(ys) / len(ys)
        center_z = sum(zs) / len(zs)

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        z_min, z_max = min(zs), max(zs)

        boundary_str = ", ".join(
            [f"({x:.1f}, {y:.1f}, {z:.1f})" for x, y, z in boundary_points[:5]]
        )
        if len(boundary_points) > 5:
            boundary_str += f", ... and {len(boundary_points) - 5} more points"

        prompt = f"""
        You are a robotics expert. Create a grinding pattern for a robotic arm.
        
        OBJECT SPECIFICATIONS:
        - Boundary points: {boundary_str}
        - Centroid: ({center_x:.1f}, {center_y:.1f}, {center_z:.1f})
        - X range: {x_min:.1f} to {x_max:.1f} mm
        - Y range: {y_min:.1f} to {y_max:.1f} mm
        - Z range: {z_min:.1f} to {z_max:.1f} mm
        
        REQUIREMENTS:
        1. Generate EXACTLY 36 points in a zigzag pattern covering the object surface
        2. Each point must be in the format: x,y,z,Rx,Ry,Rz
           - Where x,y,z are coordinates in millimeters
           - Rx,Ry,Rz are Euler angles in degrees (0-360)
        3. Start near ({x_min+10:.1f}, {y_min+10:.1f}, {z_min+5:.1f})
        4. End near ({x_max-10:.1f}, {y_max-10:.1f}, {z_max-5:.1f})
        
        EXAMPLE POINT:
        150.0,120.5,35.2,0.0,90.0,0.0
        
        YOUR RESPONSE MUST CONTAIN ONLY THE POINTS LIST WITH PATTERN_START AND PATTERN_END MARKERS:
        PATTERN_START
        x1,y1,z1,Rx1,Ry1,Rz1
        x2,y2,z2,Rx2,Ry2,Rz2
        ...
        PATTERN_END
        """

        logger.debug(f"Created explicit prompt with length: {len(prompt)}")
        return prompt

    def _create_generation_prompt(
        self,
        boundary_points: List[Tuple[float, float, float]],
        similar_patterns: List[str],
    ) -> str:
        """
        Create a prompt for the model to generate a new pattern

        Args:
            boundary_points: List of boundary point coordinates
            similar_patterns: List of similar patterns from storage

        Returns:
            Formatted prompt for the model
        """
        boundary_str = self._format_points_for_model(boundary_points)

        example_patterns = ""
        if similar_patterns:
            example_patterns = "Here are some examples of similar patterns:\n"
            for i, pattern in enumerate(similar_patterns[:3]):
                example_patterns += f"Example {i+1}:\n{pattern}\n\n"

        prompt = f"""
        Task: Generate a 3D grinding pattern for a robot arm to smooth the surface bounded by the given points.

        BOUNDARY POINTS:
        {boundary_str}
        
        INSTRUCTIONS:
        1. Create a grinding pattern with points in format (x, y, z, Rx, Ry, Rz) where:
           - x, y, z are coordinates in millimeters
           - Rx, Ry, Rz are Euler angles in degrees for end-effector orientation
        2. Generate at least 30 points for the grinding pattern
        3. Ensure the pattern covers the entire surface
        4. Orient the tool appropriately for surface grinding
        
        {example_patterns}
        
        RESPONSE FORMAT:
        You must return ONLY the list of pattern points using this exact format:
        PATTERN_START
        x1,y1,z1,Rx1,Ry1,Rz1
        x2,y2,z2,Rx2,Ry2,Rz2
        ...
        PATTERN_END
        
        DO NOT include any explanations, only the pattern points between the markers.
        """

        logger.debug(f"Generation prompt created with length: {len(prompt)}")
        return prompt

    def _parse_pattern_from_text(
        self, text: str
    ) -> List[Tuple[float, float, float, float, float, float]]:
        """
        Parse pattern points from generated text

        Args:
            text: Generated text from the model

        Returns:
            List of parsed pattern points (x, y, z, Rx, Ry, Rz)
        """
        logger.info("Parsing pattern from generated text")
        pattern_points = []

        try:
            # First try with markers
            if "PATTERN_START" in text and "PATTERN_END" in text:
                logger.info("Found PATTERN_START/END markers")
                pattern_text = (
                    text.split("PATTERN_START")[1].split("PATTERN_END")[0].strip()
                )
                lines = pattern_text.strip().split("\n")

                for line in lines:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue

                    # Parse each line into a 6D point (x, y, z, Rx, Ry, Rz)
                    try:
                        parts = line.split(",")
                        if len(parts) >= 6:
                            point = tuple(float(part.strip()) for part in parts[:6])
                            pattern_points.append(point)
                    except ValueError as e:
                        logger.warning(f"Failed to parse line '{line}': {str(e)}")

            # If no markers or no points found with markers, try alternative parsing
            if not pattern_points:
                logger.warning(
                    "No valid points found with markers, attempting alternative parsing"
                )

                # Try to find any sequences of 6 numbers separated by commas
                pattern = r"(-?\d+\.?\d*),\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*)"
                matches = re.findall(pattern, text)

                for match in matches:
                    point = tuple(float(value) for value in match[:6])
                    pattern_points.append(point)

                # If still no points, look for just numbers in parentheses
                if not pattern_points:
                    pattern = r"\(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)"
                    matches = re.findall(pattern, text)

                    for match in matches:
                        point = tuple(float(value) for value in match[:6])
                        pattern_points.append(point)

            # Log the first few points for debugging
            if pattern_points:
                logger.info(f"First few parsed points: {pattern_points[:3]}")

            # If still no points, fall back to generating a default pattern
            if not pattern_points:
                logger.warning(
                    "Could not parse any points from generated text, using fallback pattern"
                )
                pattern_points = self._generate_fallback_pattern(text)

        except Exception as e:
            logger.error(f"Error parsing pattern from text: {str(e)}")
            logger.error(traceback.format_exc())
            # Fall back to a default pattern if parsing fails
            pattern_points = self._generate_fallback_pattern(text)

        logger.info(f"Parsed {len(pattern_points)} pattern points from generated text")
        return pattern_points

    def _generate_fallback_pattern(
        self, text: str = ""
    ) -> List[Tuple[float, float, float, float, float, float]]:
        """
        Generate a fallback pattern when parsing fails

        Args:
            text: Original text from the model (for potential partial extraction)

        Returns:
            List of default pattern points
        """
        logger.info("Generating fallback pattern")

        # Create a basic zigzag pattern on a plane
        pattern_points = []

        # Start point
        x_start, y_start, z_start = 100, 100, 10
        # Step sizes
        x_step, y_step = 10, 10
        # Grid size
        rows, cols = 6, 6

        # Generate zigzag pattern
        for i in range(rows):
            row_points = []
            y = y_start + i * y_step

            for j in range(cols):
                if i % 2 == 0:
                    x = x_start + j * x_step
                else:
                    x = x_start + (cols - 1 - j) * x_step

                z = z_start
                # Default orientation (perpendicular to XY plane)
                rx, ry, rz = 0, 0, 0

                row_points.append((x, y, z, rx, ry, rz))

            pattern_points.extend(row_points)

        logger.info(f"Generated fallback pattern with {len(pattern_points)} points")
        return pattern_points

    def _points_to_str(self, points: List[Tuple]) -> str:
        """
        Convert points to a string representation for storage

        Args:
            points: List of point coordinates

        Returns:
            String representation of points
        """
        if not points:
            return "[]"

        # For 3D boundary points (x, y, z)
        if len(points[0]) == 3:
            return ";".join([f"{x:.3f},{y:.3f},{z:.3f}" for x, y, z in points])

        # For 6D pattern points (x, y, z, Rx, Ry, Rz)
        elif len(points[0]) == 6:
            return ";".join(
                [
                    f"{x:.3f},{y:.3f},{z:.3f},{rx:.3f},{ry:.3f},{rz:.3f}"
                    for x, y, z, rx, ry, rz in points
                ]
            )

        else:
            return ";".join(
                [",".join([f"{val:.3f}" for val in point]) for point in points]
            )

    def _str_to_points(self, points_str: str, dimensions: int = 3) -> List[Tuple]:
        """
        Convert string representation back to points

        Args:
            points_str: String representation of points
            dimensions: Number of dimensions per point (3 for boundary, 6 for pattern)

        Returns:
            List of point tuples
        """
        points = []

        if not points_str or points_str == "[]":
            return points

        try:
            point_strs = points_str.split(";")

            for point_str in point_strs:
                coords = point_str.split(",")

                if len(coords) >= dimensions:
                    point = tuple(float(coord) for coord in coords[:dimensions])
                    points.append(point)

        except Exception as e:
            logger.error(f"Error parsing points string: {str(e)}")
            logger.error(traceback.format_exc())

        return points

    def search_similar_boundaries(
        self, boundary_points: List[Tuple[float, float, float]], top_k: int = RAG_TOP_K
    ) -> List[Dict[str, Any]]:
        """
        Search for similar boundary patterns in local storage

        Args:
            boundary_points: List of boundary point coordinates
            top_k: Number of results to return

        Returns:
            List of dictionaries containing similar patterns
        """
        logger.info(f"Searching for {top_k} similar boundaries")

        with Timer("Similarity search", logger):
            try:
                query_embedding = self._get_embedding(boundary_points)

                results = []

                self._load_index()

                if not self.pattern_index:
                    logger.warning("No patterns found in the index")
                    return []

                for pattern_info in self.pattern_index:
                    pattern_id = pattern_info.get("id")

                    embedding_path = os.path.join(
                        self.embeddings_dir, f"{pattern_id}.pkl"
                    )
                    if not os.path.exists(embedding_path):
                        logger.warning(
                            f"Embedding file not found for pattern {pattern_id}"
                        )
                        continue

                    try:
                        with open(embedding_path, "rb") as f:
                            stored_embedding = pickle.load(f)
                    except Exception as e:
                        logger.error(
                            f"Failed to load embedding for pattern {pattern_id}: {str(e)}"
                        )
                        continue

                    similarity = 1.0 - cosine(query_embedding, stored_embedding)

                    pattern_path = os.path.join(self.patterns_dir, f"{pattern_id}.json")
                    if not os.path.exists(pattern_path):
                        logger.warning(
                            f"Pattern file not found for pattern {pattern_id}"
                        )
                        continue

                    try:
                        with open(pattern_path, "r") as f:
                            pattern_data = json.load(f)
                    except Exception as e:
                        logger.error(f"Failed to load pattern {pattern_id}: {str(e)}")
                        continue

                    boundary_str = pattern_data.get("boundary_points", "")
                    pattern_str = pattern_data.get("pattern_points", "")

                    boundary_points_stored = self._str_to_points(boundary_str, 3)
                    pattern_points_stored = self._str_to_points(pattern_str, 6)

                    results.append(
                        {
                            "id": pattern_id,
                            "similarity": similarity,
                            "distance": 1.0 - similarity,
                            "boundary_points": boundary_points_stored,
                            "pattern_points": pattern_points_stored,
                            "timestamp": pattern_data.get("timestamp", 0),
                            "raw_pattern_str": pattern_str,
                        }
                    )

                # Sort by similarity (highest first)
                results.sort(key=lambda x: x["similarity"], reverse=True)

                # Return top k results
                top_results = results[:top_k]
                logger.info(
                    f"Found {len(top_results)} similar patterns out of {len(results)} total"
                )
                return top_results

            except Exception as e:
                logger.error(f"Error searching for similar patterns: {str(e)}")
                logger.error(traceback.format_exc())
                return []

    def insert_pattern(
        self,
        boundary_points: List[Tuple[float, float, float]],
        pattern_points: List[Tuple[float, float, float, float, float, float]],
    ) -> str:
        """
        Insert a new pattern into the local storage

        Args:
            boundary_points: List of boundary point coordinates (x, y, z)
            pattern_points: List of pattern point coordinates (x, y, z, Rx, Ry, Rz)

        Returns:
            ID of the inserted pattern
        """
        logger.info(
            f"Inserting new pattern with {len(boundary_points)} boundary points "
            f"and {len(pattern_points)} pattern points"
        )

        with Timer("Pattern insertion", logger):
            try:
                # Generate embedding for the boundary points
                embedding = self._get_embedding(boundary_points)

                # Convert points to string representation for storage
                boundary_str = self._points_to_str(boundary_points)
                pattern_str = self._points_to_str(pattern_points)

                pattern_id = str(uuid.uuid4())

                pattern_data = {
                    "id": pattern_id,
                    "boundary_points": boundary_str,
                    "pattern_points": pattern_str,
                    "timestamp": int(time.time() * 1000),
                }

                # Save pattern to file
                pattern_path = os.path.join(self.patterns_dir, f"{pattern_id}.json")
                with open(pattern_path, "w") as f:
                    json.dump(pattern_data, f, indent=2)

                # Save embedding to file
                embedding_path = os.path.join(self.embeddings_dir, f"{pattern_id}.pkl")
                with open(embedding_path, "wb") as f:
                    pickle.dump(embedding, f)

                # Update index
                self._load_index()
                self.pattern_index.append(
                    {"id": pattern_id, "timestamp": pattern_data["timestamp"]}
                )
                self._save_index()

                logger.info(f"Pattern inserted with ID: {pattern_id}")
                return pattern_id

            except Exception as e:
                logger.error(f"Error inserting pattern: {str(e)}")
                logger.error(traceback.format_exc())
                raise

    def get_grinding_pattern(
        self, boundary_points: List[Tuple[float, float, float]]
    ) -> List[Tuple[float, float, float, float, float, float]]:
        """
        Main function to get a grinding pattern for a given boundary

        Args:
            boundary_points: List of boundary point coordinates (x, y, z)

        Returns:
            List of pattern point coordinates (x, y, z, Rx, Ry, Rz)
        """
        logger.info(
            f"Getting grinding pattern for boundary with {len(boundary_points)} points"
        )

        try:
            # Step 1: Search for similar boundaries
            similar_patterns = self.search_similar_boundaries(boundary_points)

            # Step 2: Generate a new pattern
            if similar_patterns:
                logger.info(
                    f"Using {len(similar_patterns)} similar patterns for generation"
                )
                pattern_points = self._generate_pattern(
                    boundary_points, similar_patterns
                )
            else:
                logger.warning(
                    "No similar patterns found, generating pattern from scratch"
                )
                pattern_points = self._generate_pattern(boundary_points, [])

            # Step 3: Insert the new pattern into storage for future use
            if pattern_points:
                try:
                    self.insert_pattern(boundary_points, pattern_points)
                except Exception as e:
                    logger.error(f"Failed to insert new pattern: {str(e)}")

            return pattern_points

        except Exception as e:
            logger.error(f"Error getting grinding pattern: {str(e)}")
            logger.error(traceback.format_exc())
            return []

    def close(self):
        """Close any open resources"""
        logger.info("Closing SurfacePatternRetriever")

    def create_example_patterns(self, num_examples: int = 10):
        """Create example patterns for testing and pre-population"""
        logger.info(f"Creating {num_examples} example patterns")

        # Clear existing data if needed
        if os.path.exists(self.patterns_dir) and os.listdir(self.patterns_dir):
            logger.info("Clearing existing pattern data")
            for file in os.listdir(self.patterns_dir):
                os.remove(os.path.join(self.patterns_dir, file))

        if os.path.exists(self.embeddings_dir) and os.listdir(self.embeddings_dir):
            for file in os.listdir(self.embeddings_dir):
                os.remove(os.path.join(self.embeddings_dir, file))

        # Reset index
        self.pattern_index = []
        self._save_index()

        example_types = ["cube", "cylinder", "sphere", "cone", "prismatic"]

        for i in range(num_examples):
            example_type = example_types[i % len(example_types)]

            # Generate boundary points based on type
            boundary_points = self._generate_example_boundary(example_type, i)

            # Generate pattern points based on boundary
            pattern_points = self._generate_example_pattern(
                boundary_points, example_type
            )

            try:
                pattern_id = self.insert_pattern(boundary_points, pattern_points)
                logger.info(
                    f"Created example pattern {i+1}/{num_examples}: {example_type} (ID: {pattern_id})"
                )
            except Exception as e:
                logger.error(f"Failed to create example pattern {i+1}: {str(e)}")

        logger.info(f"Created {len(self.pattern_index)} example patterns")

    def _generate_example_boundary(
        self, shape_type: str, index: int
    ) -> List[Tuple[float, float, float]]:
        """
        Generate example boundary points for a given shape type

        Args:
            shape_type: Type of shape to generate
            index: Index for variation

        Returns:
            List of 3D boundary points
        """
        np.random.seed(index)  # For reproducibility

        size = 100 + np.random.rand() * 200
        center_x = np.random.rand() * 500
        center_y = np.random.rand() * 500
        center_z = np.random.rand() * 200

        points = []

        if shape_type == "cube":
            half_size = size / 2
            for x_sign in [-1, 1]:
                for y_sign in [-1, 1]:
                    for z_sign in [-1, 1]:
                        x = center_x + x_sign * half_size
                        y = center_y + y_sign * half_size
                        z = center_z + z_sign * half_size
                        points.append((x, y, z))

        elif shape_type == "cylinder":
            radius = size / 2
            height = size
            num_points = 8
            for i in range(num_points):
                angle = 2 * np.pi * i / num_points
                x = center_x + radius * np.cos(angle)
                y = center_y + radius * np.sin(angle)

                points.append((x, y, center_z - height / 2))
                points.append((x, y, center_z + height / 2))

        elif shape_type == "sphere":
            radius = size / 2
            num_points = 12
            for i in range(num_points):
                phi = np.pi * i / (num_points - 1)
                for j in range(num_points):
                    theta = 2 * np.pi * j / num_points

                    x = center_x + radius * np.sin(phi) * np.cos(theta)
                    y = center_y + radius * np.sin(phi) * np.sin(theta)
                    z = center_z + radius * np.cos(phi)

                    points.append((x, y, z))

        elif shape_type == "cone":
            radius = size / 2
            height = size
            num_points = 8

            points.append((center_x, center_y, center_z + height / 2))

            for i in range(num_points):
                angle = 2 * np.pi * i / num_points
                x = center_x + radius * np.cos(angle)
                y = center_y + radius * np.sin(angle)
                z = center_z - height / 2
                points.append((x, y, z))

        elif shape_type == "prismatic":
            width = size
            depth = size * 0.6
            height = size * 1.5

            for x_sign in [-1, 1]:
                for y_sign in [-1, 1]:
                    for z_sign in [-1, 1]:
                        x = center_x + x_sign * width / 2
                        y = center_y + y_sign * depth / 2
                        z = center_z + z_sign * height / 2
                        points.append((x, y, z))

        else:
            # Default: generate a random cloud of points
            num_points = 20
            for _ in range(num_points):
                x = center_x + (np.random.rand() - 0.5) * size
                y = center_y + (np.random.rand() - 0.5) * size
                z = center_z + (np.random.rand() - 0.5) * size
                points.append((x, y, z))

        return points

    def _generate_example_pattern(
        self, boundary_points: List[Tuple[float, float, float]], shape_type: str
    ) -> List[Tuple[float, float, float, float, float, float]]:
        """
        Generate example pattern points for given boundary

        Args:
            boundary_points: Boundary points defining the shape
            shape_type: Type of shape

        Returns:
            List of 6D pattern points (x, y, z, Rx, Ry, Rz)
        """
        pattern_points = []

        if not boundary_points:
            return []

        xs, ys, zs = zip(*boundary_points)
        center_x = sum(xs) / len(xs)
        center_y = sum(ys) / len(ys)
        center_z = sum(zs) / len(zs)

        radius = max(
            max(abs(x - center_x) for x in xs),
            max(abs(y - center_y) for y in ys),
            max(abs(z - center_z) for z in zs),
        )

        num_points = 50 + int(np.random.rand() * 50)  # Between 50-100 points

        if shape_type == "cube":
            faces = [
                ("x", 1),
                ("x", -1),
                ("y", 1),
                ("y", -1),
                ("z", 1),
                ("z", -1),
            ]

            for axis, direction in faces:
                for i in range(num_points // 6):
                    t = i / (num_points / 6 - 1)
                    spiral_r = radius * (1 - t * 0.5)
                    angle = t * 4 * np.pi

                    if axis == "x":
                        x = center_x + direction * radius
                        y = center_y + spiral_r * np.cos(angle)
                        z = center_z + spiral_r * np.sin(angle)
                        rx, ry, rz = 0, 90 * direction, 0

                    elif axis == "y":
                        x = center_x + spiral_r * np.cos(angle)
                        y = center_y + direction * radius
                        z = center_z + spiral_r * np.sin(angle)
                        rx, ry, rz = 90 * direction, 0, 0

                    else:
                        x = center_x + spiral_r * np.cos(angle)
                        y = center_y + spiral_r * np.sin(angle)
                        z = center_z + direction * radius
                        rx, ry, rz = 0, 0, 180 if direction < 0 else 0

                    pattern_points.append((x, y, z, rx, ry, rz))

        elif shape_type == "cylinder":
            # Generate a spiral pattern on the curved surface
            height_points = int(np.sqrt(num_points))
            angle_points = num_points // height_points

            for i in range(height_points):
                h_t = i / (height_points - 1)
                z = center_z - radius + h_t * 2 * radius

                for j in range(angle_points):
                    a_t = j / angle_points
                    angle = a_t * 2 * np.pi + h_t * 4 * np.pi

                    x = center_x + radius * np.cos(angle)
                    y = center_y + radius * np.sin(angle)

                    rx = 0
                    ry = np.degrees(np.arctan2(y - center_y, x - center_x))
                    rz = 0

                    pattern_points.append((x, y, z, rx, ry, rz))

        elif shape_type in ["sphere", "cone"]:
            # Generate a spiral pattern covering the surface
            num_spirals = 5
            points_per_spiral = num_points // num_spirals

            for spiral in range(num_spirals):
                spiral_offset = spiral * 2 * np.pi / num_spirals

                for i in range(points_per_spiral):
                    t = i / (points_per_spiral - 1)  # 0 to 1

                    # Parametric equation for spiral on a sphere
                    phi = np.pi * t  # 0 to pi (top to bottom)
                    theta = 8 * np.pi * t + spiral_offset  # Many rotations + offset

                    x = center_x + radius * np.sin(phi) * np.cos(theta)
                    y = center_y + radius * np.sin(phi) * np.sin(theta)
                    z = center_z + radius * np.cos(phi)

                    # Orient perpendicular to the surface (pointing out from center)
                    r_x = np.degrees(phi) - 90  # Pitch
                    r_y = 0
                    r_z = np.degrees(theta)  # Yaw

                    pattern_points.append((x, y, z, r_x, r_y, r_z))

        elif shape_type == "prismatic":
            # Similar to cube but with different dimensions
            faces = [
                ("x", 1),
                ("x", -1),  # +x and -x faces
                ("y", 1),
                ("y", -1),
                ("z", 1),
                ("z", -1),
            ]

            for axis, direction in faces:
                face_points = []
                width = radius * 2 if axis != "x" else radius * 2
                height = radius * 2 if axis != "y" else radius * 1.2

                for i in range(num_points // 6):
                    t = i / (num_points / 6 - 1)

                    if axis == "x":
                        rows = 10
                        row = int(t * rows)
                        within_row = (t * rows) % 1

                        x = center_x + direction * radius
                        y = center_y - height / 2 + row * height / rows

                        # Zigzag within row
                        if row % 2 == 0:  # Even rows go left to right
                            z = center_z - width / 2 + within_row * width
                        else:  # Odd rows go right to left
                            z = center_z + width / 2 - within_row * width

                        rx, ry, rz = 0, 90 * direction, 0

                    elif axis == "y":
                        rows = 10
                        row = int(t * rows)
                        within_row = (t * rows) % 1

                        x = center_x - width / 2 + row * width / rows
                        y = center_y + direction * radius * 0.6

                        if row % 2 == 0:
                            z = center_z - height / 2 + within_row * height
                        else:
                            z = center_z + height / 2 - within_row * height

                        rx, ry, rz = 90 * direction, 0, 0

                    else:
                        rows = 10
                        row = int(t * rows)
                        within_row = (t * rows) % 1

                        z = center_z + direction * radius * 1.5
                        x = center_x - width / 2 + row * width / rows

                        if row % 2 == 0:
                            y = center_y - height / 2 + within_row * height
                        else:
                            y = center_y + height / 2 - within_row * height

                        rx, ry, rz = 0, 0, 180 if direction < 0 else 0

                    face_points.append((x, y, z, rx, ry, rz))

                pattern_points.extend(face_points)

        else:
            # Random pattern for unknown shapes
            for _ in range(num_points):
                # Random point on or near the surface
                phi = np.random.rand() * np.pi
                theta = np.random.rand() * 2 * np.pi
                r = radius * (0.8 + 0.2 * np.random.rand())

                x = center_x + r * np.sin(phi) * np.cos(theta)
                y = center_y + r * np.sin(phi) * np.sin(theta)
                z = center_z + r * np.cos(phi)

                # Random orientation, but somewhat facing inward
                toward_center = np.array([center_x - x, center_y - y, center_z - z])
                toward_center = toward_center / np.linalg.norm(toward_center)

                # Convert to Euler angles (approximate)
                rx = np.degrees(np.arcsin(toward_center[2]))
                ry = np.degrees(np.arctan2(toward_center[1], toward_center[0]))
                rz = np.random.rand() * 360

                pattern_points.append((x, y, z, rx, ry, rz))

        return pattern_points


if __name__ == "__main__":
    try:
        logger.info("Starting Surface Pattern RAG integration")

        retriever = SurfacePatternRetriever()

        # Check if examples should be created
        create_examples = False

        if create_examples:
            logger.info("Creating example patterns for database")
            retriever.create_example_patterns(num_examples=10)

        test_boundary = [
            (100.0, 100.0, 0.0),
            (100.0, 200.0, 0.0),
            (200.0, 200.0, 0.0),
            (200.0, 100.0, 0.0),
            (100.0, 100.0, 10.0),
            (100.0, 200.0, 10.0),
            (200.0, 200.0, 10.0),
            (200.0, 100.0, 10.0),
        ]

        logger.info("Test boundary points created")

        pattern = retriever.get_grinding_pattern(test_boundary)

        logger.info(f"Retrieved grinding pattern with {len(pattern)} points")

        # Print the pattern in a nicely formatted way
        print("\n===== GENERATED GRINDING PATTERN =====")
        print(f"Total points: {len(pattern)}")
        print("\nSample points (first 10):")
        for i, point in enumerate(pattern[:10]):
            x, y, z, rx, ry, rz = point
            print(
                f"Point {i+1:2d}: ({x:7.2f}, {y:7.2f}, {z:7.2f}) - Orientation: ({rx:7.2f}, {ry:7.2f}, {rz:7.2f})"
            )

        if len(pattern) > 10:
            print(f"\n... and {len(pattern) - 10} more points")

        # Save pattern to file for inspection
        pattern_file = os.path.join(DATA_DIR, "latest_pattern.json")
        with open(pattern_file, "w") as f:
            json.dump(
                {
                    "boundary_points": [list(p) for p in test_boundary],
                    "pattern_points": [list(p) for p in pattern],
                    "timestamp": int(time.time() * 1000),
                },
                f,
                indent=2,
            )

        print(f"\nPattern saved to {pattern_file}")
        print("=====================================\n")

        retriever.close()

        logger.info("Surface Pattern RAG integration complete")

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        logger.error(traceback.format_exc())
