"""
Automated Conveyor Belt Sorting Engine
Simulates pneumatic air jet actuator signals, bin routing, and environmental sustainability calculations.
"""

import time
from config import (
    CLASS_INFO, CONFIDENCE_THRESHOLD, CONVEYOR_BELT_SPEED_MPS,
    CAMERA_DISTANCE_TO_JETS_M, AIR_JET_PULSE_MS, AIR_JET_PRESSURE_BAR
)

class AutomatedSortingSystem:
    """Simulates automated robotic/pneumatic waste sorting hardware system."""
    
    def __init__(self, conveyor_speed=CONVEYOR_BELT_SPEED_MPS, camera_distance=CAMERA_DISTANCE_TO_JETS_M):
        self.conveyor_speed = conveyor_speed
        self.camera_distance = camera_distance
        # Travel time from camera sensor to pneumatic air jet array (in milliseconds)
        self.travel_delay_ms = int((self.camera_distance / self.conveyor_speed) * 1000)
        self.total_processed = 0
        self.bin_counts = {c: 0 for c in CLASS_INFO.keys()}
        self.total_co2_saved_kg = 0.0
        self.total_energy_saved_kwh = 0.0

    def process_item(self, predicted_class, confidence, item_weight_kg=0.25):
        """
        Processes a classified waste material item passing along the conveyor belt.
        
        Args:
            predicted_class (str): Predicted waste category (cardboard, glass, metal, paper, plastic, trash).
            confidence (float): EfficientNetB0 output confidence (0.0 to 1.0).
            item_weight_kg (float): Estimated weight of item in kilograms.
            
        Returns:
            dict: Hardware control signal & environmental impact breakdown.
        """
        info = CLASS_INFO.get(predicted_class, CLASS_INFO["trash"])
        self.total_processed += 1
        
        # Check if confidence meets operational threshold
        if confidence >= CONFIDENCE_THRESHOLD and info["recyclable"]:
            action = "PNEUMATIC_EJECT"
            status = "AUTOMATED_SORTED"
            jet_id = info["pneumatic_jet_id"]
            bin_name = info["bin"]
            stream = info["stream"]
            
            # Sustainability impact
            co2_saved = info["co2_saved_per_kg"] * item_weight_kg
            energy_saved = info["energy_saved_kwh"] * item_weight_kg
            
            self.bin_counts[predicted_class] += 1
            self.total_co2_saved_kg += co2_saved
            self.total_energy_saved_kwh += energy_saved
        elif confidence < CONFIDENCE_THRESHOLD:
            action = "MANUAL_INSPECTION_DIVERT"
            status = "LOW_CONFIDENCE_REJECT"
            jet_id = 0  # Pass to manual quality check chute
            bin_name = "Chute 0 - Quality Inspection"
            stream = "Manual Audit Stream"
            co2_saved = 0.0
            energy_saved = 0.0
        else:
            # Trash / non-recyclable residual
            action = "CONTINUE_TO_RESIDUAL"
            status = "RESIDUAL_LANDFILL"
            jet_id = info["pneumatic_jet_id"]
            bin_name = info["bin"]
            stream = info["stream"]
            co2_saved = 0.0
            energy_saved = 0.0
            self.bin_counts["trash"] += 1

        sorting_result = {
            "item_id": f"ITEM-{self.total_processed:05d}",
            "material": predicted_class.capitalize(),
            "confidence": float(confidence),
            "confidence_percent": f"{confidence * 100:.1f}%",
            "status": status,
            "action": action,
            "target_bin": bin_name,
            "target_stream": stream,
            "hardware_signals": {
                "pneumatic_jet_id": jet_id,
                "conveyor_speed_mps": self.conveyor_speed,
                "trigger_delay_ms": self.travel_delay_ms,
                "air_pulse_duration_ms": AIR_JET_PULSE_MS,
                "air_pressure_bar": AIR_JET_PRESSURE_BAR,
            },
            "environmental_impact": {
                "co2_saved_kg": round(co2_saved, 3),
                "energy_saved_kwh": round(energy_saved, 3),
                "weight_kg": item_weight_kg
            }
        }
        return sorting_result

    def get_system_summary(self):
        """Returns cumulative sorting statistics and sustainability metrics."""
        return {
            "total_items_processed": self.total_processed,
            "bin_breakdown": self.bin_counts,
            "total_co2_saved_kg": round(self.total_co2_saved_kg, 2),
            "total_energy_saved_kwh": round(self.total_energy_saved_kwh, 2),
            "recycling_purity_rate": f"{((self.total_processed - self.bin_counts['trash']) / max(1, self.total_processed)) * 100:.1f}%"
        }
