"""
Wildlife Sighting Simulator — Demo Mode for Testing
Generates realistic wildlife sightings with GPS coordinates for testing the BigV app.
"""

import random
import time
from datetime import datetime, timezone
from typing import Optional

# Safari park locations with realistic coordinates
SAFARI_PARKS = {
    "serengeti": {
        "name": "Serengeti National Park, Tanzania",
        "center": (-2.3333, 34.8333),
        "radius": 0.5,  # degrees (~55km)
        "animals": ["lion", "leopard", "cheetah", "elephant", "buffalo", "wildebeest", "zebra", "giraffe", "hippopotamus"],
    },
    "maasai_mara": {
        "name": "Maasai Mara, Kenya",
        "center": (-1.5000, 35.1500),
        "radius": 0.3,
        "animals": ["lion", "leopard", "cheetah", "elephant", "buffalo", "wildebeest", "zebra", "giraffe"],
    },
    "kruger": {
        "name": "Kruger National Park, South Africa",
        "center": (-23.9884, 31.5547),
        "radius": 0.8,
        "animals": ["lion", "leopard", "cheetah", "elephant", "buffalo", "rhinoceros", "giraffe", "zebra", "hippopotamus"],
    },
    "okavango": {
        "name": "Okavango Delta, Botswana",
        "center": (-19.3000, 22.9000),
        "radius": 0.4,
        "animals": ["lion", "leopard", "elephant", "hippopotamus", "crocodile", "zebra", "giraffe"],
    },
}

# Animal behavior patterns (affects detection probability)
ANIMAL_PATTERNS = {
    "lion": {"time_preference": "dawn_dusk", "group_size": (1, 5), "confidence_range": (0.75, 0.95)},
    "leopard": {"time_preference": "night", "group_size": (1, 1), "confidence_range": (0.65, 0.90)},
    "cheetah": {"time_preference": "day", "group_size": (1, 3), "confidence_range": (0.70, 0.92)},
    "elephant": {"time_preference": "any", "group_size": (3, 15), "confidence_range": (0.85, 0.98)},
    "buffalo": {"time_preference": "any", "group_size": (5, 50), "confidence_range": (0.80, 0.95)},
    "rhinoceros": {"time_preference": "dawn_dusk", "group_size": (1, 2), "confidence_range": (0.70, 0.88)},
    "giraffe": {"time_preference": "day", "group_size": (2, 8), "confidence_range": (0.82, 0.96)},
    "zebra": {"time_preference": "day", "group_size": (5, 30), "confidence_range": (0.85, 0.97)},
    "wildebeest": {"time_preference": "day", "group_size": (10, 100), "confidence_range": (0.80, 0.95)},
    "hippopotamus": {"time_preference": "dawn_dusk", "group_size": (3, 12), "confidence_range": (0.78, 0.93)},
    "crocodile": {"time_preference": "any", "group_size": (1, 5), "confidence_range": (0.72, 0.89)},
}

ALERT_PRIORITY = {
    "lion": "critical",
    "leopard": "critical",
    "cheetah": "critical",
    "rhinoceros": "high",
    "hippopotamus": "high",
    "elephant": "high",
    "buffalo": "high",
    "crocodile": "critical",
    "default": "low",
}


class WildlifeSimulator:
    """Generates realistic wildlife sightings for testing."""
    
    def __init__(self, park: str = "serengeti"):
        self.park = park
        self.park_data = SAFARI_PARKS.get(park, SAFARI_PARKS["serengeti"])
        self.sighting_count = 0
    
    def generate_gps(self) -> tuple[float, float]:
        """Generate random GPS coordinates within park boundaries."""
        center_lat, center_lng = self.park_data["center"]
        radius = self.park_data["radius"]
        
        # Random offset within radius
        lat_offset = random.uniform(-radius, radius)
        lng_offset = random.uniform(-radius, radius)
        
        return (
            round(center_lat + lat_offset, 6),
            round(center_lng + lng_offset, 6)
        )
    
    def select_animal(self) -> str:
        """Select a random animal based on park wildlife."""
        animals = self.park_data["animals"]
        
        # Weight rare animals lower
        weights = []
        for animal in animals:
            if animal in ["rhinoceros", "leopard"]:
                weights.append(0.1)  # Rare
            elif animal in ["lion", "cheetah"]:
                weights.append(0.3)  # Uncommon
            else:
                weights.append(1.0)  # Common
        
        return random.choices(animals, weights=weights)[0]
    
    def generate_sighting(self) -> dict:
        """Generate a complete wildlife sighting."""
        animal = self.select_animal()
        pattern = ANIMAL_PATTERNS.get(animal, {"group_size": (1, 1), "confidence_range": (0.7, 0.9)})
        
        lat, lng = self.generate_gps()
        confidence = round(random.uniform(*pattern["confidence_range"]), 3)
        group_size = random.randint(*pattern["group_size"])
        
        animals = []
        for i in range(group_size):
            conf_variation = random.uniform(-0.05, 0.05)
            animals.append({
                "label": animal,
                "confidence": round(max(0.5, min(1.0, confidence + conf_variation)), 3),
                "bbox": {
                    "x1": random.randint(100, 400),
                    "y1": random.randint(100, 300),
                    "x2": random.randint(500, 800),
                    "y2": random.randint(400, 600),
                },
                "priority": ALERT_PRIORITY.get(animal, "low"),
            })
        
        self.sighting_count += 1
        
        return {
            "id": f"sim-{self.sighting_count}-{int(time.time())}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "animals": animals,
            "location": self.park_data["name"],
            "camera_id": f"SIM-CAM-{random.randint(1, 10):02d}",
            "confidence": confidence,
            "alert_level": self._calculate_alert_level(confidence, animal),
            "lat": lat,
            "lng": lng,
            "simulated": True,
        }
    
    def _calculate_alert_level(self, confidence: float, animal: str) -> str:
        """Calculate alert level based on confidence and animal type."""
        priority = ALERT_PRIORITY.get(animal, "low")
        
        if priority == "critical" and confidence >= 0.85:
            return "critical"
        elif priority == "critical" or (priority == "high" and confidence >= 0.80):
            return "high"
        elif confidence >= 0.70:
            return "medium"
        else:
            return "low"
    
    def generate_batch(self, count: int = 5) -> list[dict]:
        """Generate multiple sightings."""
        return [self.generate_sighting() for _ in range(count)]


def simulate_safari_drive(park: str = "serengeti", duration_minutes: int = 30, sightings_per_hour: int = 8):
    """
    Simulate a complete safari drive with realistic timing.
    
    Args:
        park: Safari park name
        duration_minutes: Drive duration in minutes
        sightings_per_hour: Average sightings per hour
    
    Yields:
        Wildlife sighting dictionaries
    """
    simulator = WildlifeSimulator(park)
    interval = 60 / sightings_per_hour  # minutes between sightings
    
    elapsed = 0
    while elapsed < duration_minutes:
        yield simulator.generate_sighting()
        
        # Random interval with some variation
        wait = interval * random.uniform(0.5, 1.5)
        elapsed += wait
        time.sleep(wait * 60)  # Convert to seconds


# CLI for testing
if __name__ == "__main__":
    import sys
    
    park = sys.argv[1] if len(sys.argv) > 1 else "serengeti"
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    print(f"🦁 BigV Wildlife Simulator — {park.upper()}")
    print(f"Generating {count} sightings...\n")
    
    simulator = WildlifeSimulator(park)
    sightings = simulator.generate_batch(count)
    
    for i, sighting in enumerate(sightings, 1):
        animals = ", ".join([f"{a['label']} ({a['confidence']:.0%})" for a in sighting["animals"]])
        print(f"{i}. {sighting['alert_level'].upper():8} | {animals:40} | GPS: {sighting['lat']:.4f}, {sighting['lng']:.4f}")
    
    print(f"\n✅ Generated {len(sightings)} simulated sightings")

# Made with Bob
