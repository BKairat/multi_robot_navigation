from __future__ import annotations
import numpy as np

def rotation(points: np.array, angle: float, ref_point: np.array | None = None) -> np.array:
    if points.shape == (2,):
        if ref_point is None:
            ref_point = np.zeros(2)
        
        rotated_points = points - ref_point
        rotated_points = rotated_points @ np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle),  np.cos(angle)]
        ]) 
        rotated_points += ref_point
    else:
        pos = np.mean(points, axis=0)
        if ref_point is None:
            ref_point = pos
            
        rotated_points = points - ref_point
        rotated_points = rotated_points @ np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle),  np.cos(angle)]
        ])
        rotated_points += pos
    return rotated_points

def tarnslation(points: np.array, translation: np.array) -> np.array:
    translated_points = points + translation
    return translated_points

def set_to(points: np.array, translation: np.array, angle: float) -> np.array:
    points = np.copy(points)
    points -= np.mean(points, axis=0)
    points = rotation(points, angle)
    points += translation
    return points

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_vector(vector1):
    unit_vec = unit_vector(vector1)
    return np.arctan2(unit_vec[0], unit_vec[1])

def normalize_angle(angle):
    normalized_angle = np.mod(angle + np.pi, 2 * np.pi) - np.pi
    return normalized_angle
