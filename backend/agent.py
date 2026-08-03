"""
Wildlife Agent — Agentic AI layer for Safari Guide & Tourist use.
Provides safari-specific summaries, Big Five tracking, and guide-quality recommendations.
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from datetime import datetime
from typing import Optional

import numpy as np

from detector import WildlifeDetector, ALERT_PRIORITY

logger = logging.getLogger("wildlife_spotter.agent")

MEMORY_WINDOW = 20

# ─── Big Five ─────────────────────────────────────────────────────────────────
BIG_FIVE = {"lion", "leopard", "elephant", "rhinoceros", "buffalo"}

# ─── Safari animal knowledge base ─────────────────────────────────────────────
ANIMAL_INFO = {
    "lion": {
        "swahili": "Simba",
        "habitat": "Open savanna, grasslands, woodland edges",
        "diet": "Carnivore — wildebeest, zebra, buffalo",
        "behavior": "Social; lives in prides of 10–15. Most active at dawn and dusk.",
        "photo_tip": "Use a 300mm+ lens. Stay in vehicle. Best light: golden hour.",
        "big_five": True,
    },
    "leopard": {
        "swahili": "Chui",
        "habitat": "Dense bush, riverine forest, rocky outcrops",
        "diet": "Carnivore — impala, baboon, small antelope",
        "behavior": "Solitary and nocturnal. Often rests in trees with kills.",
        "photo_tip": "Scan tree branches carefully. Patience required — rare sighting!",
        "big_five": True,
    },
    "elephant": {
        "swahili": "Tembo",
        "habitat": "Savanna, forest, bushveld",
        "diet": "Herbivore — up to 300kg of vegetation per day",
        "behavior": "Highly intelligent, matriarchal herds. Excellent memory.",
        "photo_tip": "Wide angle for herds. Watch for mock charges — back away slowly.",
        "big_five": True,
    },
    "rhinoceros": {
        "swahili": "Kifaru",
        "habitat": "Grassland, savanna, bushveld",
        "diet": "Herbivore — grass (white rhino), leaves/shrubs (black rhino)",
        "behavior": "Mostly solitary. Poor eyesight but excellent hearing and smell.",
        "photo_tip": "Extremely rare — report to ranger immediately. Stay 100m+ away.",
        "big_five": True,
    },
    "buffalo": {
        "swahili": "Nyati / Mbogo",
        "habitat": "Savanna, floodplains, forest edges near water",
        "diet": "Herbivore — grass",
        "behavior": "Large herds (100–1000). Old bulls ('dagga boys') are unpredictable.",
        "photo_tip": "Shoot from vehicle only. Old bulls are considered most dangerous.",
        "big_five": True,
    },
    "cheetah": {
        "swahili": "Duma",
        "habitat": "Open grassland, semi-arid savanna",
        "diet": "Carnivore — gazelle, impala, hare",
        "behavior": "Diurnal hunter. Fastest land animal (110 km/h). Solitary or coalitions.",
        "photo_tip": "Best in open plains. Shoot during hunts at dawn.",
        "big_five": False,
    },
    "giraffe": {
        "swahili": "Twiga",
        "habitat": "Open woodland, savanna",
        "diet": "Herbivore — acacia leaves",
        "behavior": "Tallest animal. Vulnerable when drinking. Gentle unless threatened.",
        "photo_tip": "Silhouette against sunset sky. Wide angle for full body.",
        "big_five": False,
    },
    "zebra": {
        "swahili": "Punda Milia",
        "habitat": "Open grassland, savanna",
        "diet": "Herbivore — grass",
        "behavior": "Highly social, large herds. Each stripe pattern is unique.",
        "photo_tip": "Black & white contrast — shoot in shade for best detail.",
        "big_five": False,
    },
    "hippopotamus": {
        "swahili": "Kiboko",
        "habitat": "Rivers, lakes, wetlands",
        "diet": "Herbivore — grass (grazes at night)",
        "behavior": "Most dangerous animal in Africa. Territorial in water.",
        "photo_tip": "Shoot from riverbank at safe distance. Dawn/dusk for yawning shots.",
        "big_five": False,
    },
    "crocodile": {
        "swahili": "Mamba",
        "habitat": "Rivers, lakes, estuaries",
        "diet": "Carnivore — fish, wildebeest, zebra at crossings",
        "behavior": "Ambush predator. Can remain motionless for hours.",
        "photo_tip": "Telephoto from bank. Never approach water's edge on foot.",
        "big_five": False,
    },
    "wildebeest": {
        "swahili": "Nyumbu",
        "habitat": "Open grassland, savanna",
        "diet": "Herbivore — grass",
        "behavior": "Famous for Great Migration (1.5M animals). Calving Jan–Feb.",
        "photo_tip": "River crossings are iconic — use fast shutter speed (1/1000s+).",
        "big_five": False,
    },
    "chimpanzee": {
        "swahili": "Sokwe",
        "habitat": "Tropical forest",
        "diet": "Omnivore — fruit, insects, small animals",
        "behavior": "Highly intelligent. Tool use observed. Complex social hierarchy.",
        "photo_tip": "Forest light is tricky — use high ISO. Move slowly and quietly.",
        "big_five": False,
    },
    "gorilla": {
        "swahili": "Gorila",
        "habitat": "Mountain and lowland forest",
        "diet": "Herbivore — leaves, stems, fruit",
        "behavior": "Gentle giants unless threatened. Silverback leads family group.",
        "photo_tip": "Gorilla trekking permit required. No flash. Stay 7m minimum.",
        "big_five": False,
    },
}

# ─── Extended African species fact cards ──────────────────────────────────────
# Covers animals the COCO remap and Roboflow model can now correctly identify
ANIMAL_INFO.update({
    "antelope": {
        "swahili": "Swala / Paa",
        "habitat": "Open savanna, grassland, bushveld",
        "diet": "Herbivore — grass, leaves, shrubs",
        "behavior": "Highly alert, fast runners. Form large herds for protection.",
        "photo_tip": "Fast shutter speed (1/1000s+). Shoot at eye level for best impact.",
        "big_five": False,
    },
    "impala": {
        "swahili": "Swala Pala",
        "habitat": "Open woodland, savanna near water",
        "diet": "Herbivore — grass and browse",
        "behavior": "Most common antelope. Males territorial. Explosive leaps when alarmed.",
        "photo_tip": "Golden hour light brings out their reddish coat beautifully.",
        "big_five": False,
    },
    "gazelle": {
        "swahili": "Swala",
        "habitat": "Open grassland, semi-arid savanna",
        "diet": "Herbivore — grass, leaves",
        "behavior": "Famous for 'stotting' (high leaping) to signal fitness to predators.",
        "photo_tip": "Include the golden grass for context. Use burst mode during sprints.",
        "big_five": False,
    },
    "springbok": {
        "swahili": "Springboko",
        "habitat": "Open semi-arid savanna, Kalahari",
        "diet": "Herbivore — grass and browse",
        "behavior": "National animal of South Africa. Pronks (leaps) when excited.",
        "photo_tip": "Capture the pronking leap — use high burst rate.",
        "big_five": False,
    },
    "kudu": {
        "swahili": "Tandala",
        "habitat": "Woodland, bushveld, rocky hills",
        "diet": "Herbivore — leaves, fruit, grass",
        "behavior": "Males have spectacular spiral horns. Excellent jumpers.",
        "photo_tip": "Frame the spiral horns against the sky for a dramatic shot.",
        "big_five": False,
    },
    "warthog": {
        "swahili": "Ngiri",
        "habitat": "Open grassland, savanna",
        "diet": "Herbivore — roots, grass, tubers",
        "behavior": "Runs with tail upright. Kneels to graze. Lions' favourite snack.",
        "photo_tip": "Low angle shot emphasises their comical tusks and warts.",
        "big_five": False,
    },
    "wild dog": {
        "swahili": "Mbwa Mwitu",
        "habitat": "Open savanna, woodland",
        "diet": "Carnivore — impala, wildebeest, gazelle",
        "behavior": "Most successful African predator (~80% hunt success). Endangered.",
        "photo_tip": "Extremely rare — photograph immediately. Report to rangers.",
        "big_five": False,
    },
    "hyena": {
        "swahili": "Fisi",
        "habitat": "Savanna, grassland, woodland",
        "diet": "Carnivore/scavenger — kills own prey and steals from lions",
        "behavior": "Highly intelligent. Complex social clans. Powerful bone-crushing jaws.",
        "photo_tip": "Dawn/dusk best. Capture their spotted coat in low golden light.",
        "big_five": False,
    },
    "ostrich": {
        "swahili": "Mbuni",
        "habitat": "Open savanna, semi-arid grassland",
        "diet": "Omnivore — seeds, plants, insects",
        "behavior": "Largest bird, cannot fly. Fastest running bird (70 km/h). Powerful kick.",
        "photo_tip": "Wide angle to show scale. Males have striking black and white plumage.",
        "big_five": False,
    },
    "jackal": {
        "swahili": "Bweha",
        "habitat": "Savanna, open woodland",
        "diet": "Omnivore — small animals, fruit, carrion",
        "behavior": "Monogamous pairs. Often scavenges near lion kills.",
        "photo_tip": "Capture at kill sites — fascinating interaction with vultures.",
        "big_five": False,
    },
    "spotted hyena": {
        "swahili": "Fisi Madoa",
        "habitat": "Savanna, grassland, woodland",
        "diet": "Carnivore/scavenger — Africa's most successful predator",
        "behavior": "Matriarchal clans. Kills 95% of own prey despite scavenger reputation.",
        "photo_tip": "Dawn/dusk best. Spotted coat glows in low golden light.",
        "big_five": False,
    },
    "african lion": {
        "swahili": "Simba",
        "habitat": "Open savanna, grasslands, woodland edges",
        "diet": "Carnivore — wildebeest, zebra, buffalo",
        "behavior": "Social; lives in prides of 10–15. Most active at dawn and dusk.",
        "photo_tip": "Use a 300mm+ lens. Stay in vehicle. Best light: golden hour.",
        "big_five": False,
    },
    "african elephant": {
        "swahili": "Tembo",
        "habitat": "Savanna, forest, bushveld",
        "diet": "Herbivore — up to 300kg of vegetation per day",
        "behavior": "Highly intelligent, matriarchal herds. Excellent memory.",
        "photo_tip": "Wide angle for herds. Watch for mock charges — back away slowly.",
        "big_five": True,
    },
    "black rhinoceros": {
        "swahili": "Kifaru Mdomo-Ncha",
        "habitat": "Bushveld, shrubland, semi-arid savanna",
        "diet": "Herbivore — leaves, shrubs, branches (browser)",
        "behavior": "Solitary, highly territorial. Poor eyesight, charges on sound/smell.",
        "photo_tip": "Critically endangered — report GPS to ranger immediately. Stay 200m+.",
        "big_five": True,
    },
    "african buffalo": {
        "swahili": "Nyati",
        "habitat": "Savanna, floodplains, forest edges near water",
        "diet": "Herbivore — grass",
        "behavior": "Large herds (100–1000). Lone bulls ('dagga boys') are most dangerous.",
        "photo_tip": "Shoot from vehicle only. Low angle emphasises the massive horns.",
        "big_five": True,
    },
    "plains zebra": {
        "swahili": "Punda Milia",
        "habitat": "Open grassland, savanna",
        "diet": "Herbivore — grass",
        "behavior": "Highly social. Each stripe pattern unique. Migrate with wildebeest.",
        "photo_tip": "Black & white contrast — overcast light gives best stripe detail.",
        "big_five": False,
    },
    "reticulated giraffe": {
        "swahili": "Twiga",
        "habitat": "Open woodland, savanna, bushland",
        "diet": "Herbivore — acacia leaves",
        "behavior": "Tallest animal on earth. Vulnerable when drinking. Gentle giant.",
        "photo_tip": "Silhouette against sunset sky. Wide angle for full body.",
        "big_five": False,
    },
    "topi": {
        "swahili": "Nyamera / Sassaby",
        "habitat": "Open floodplain grassland, moist savanna",
        "diet": "Herbivore — grass",
        "behavior": "One of fastest antelopes (70 km/h). Males stand on termite mounds.",
        "photo_tip": "Capture males on termite mounds — dramatic silhouette shots.",
        "big_five": False,
    },
    "thomson gazelle": {
        "swahili": "Swala Tomi",
        "habitat": "Open short-grass plains",
        "diet": "Herbivore — short grass",
        "behavior": "Famous for 'stotting'. Cheetah's primary prey. Very fast (80 km/h).",
        "photo_tip": "Use burst mode — capture mid-stott leap for iconic shot.",
        "big_five": False,
    },
    "thomson's gazelle": {
        "swahili": "Swala Tomi",
        "habitat": "Open short-grass plains",
        "diet": "Herbivore — short grass",
        "behavior": "Famous for 'stotting'. Cheetah's primary prey. Very fast (80 km/h).",
        "photo_tip": "Use burst mode — capture mid-stott leap for iconic shot.",
        "big_five": False,
    },
    "thomsons gazelle": {
        "swahili": "Swala Tomi",
        "habitat": "Open short-grass plains",
        "diet": "Herbivore — short grass",
        "behavior": "Famous for 'stotting'. Cheetah's primary prey. Very fast (80 km/h).",
        "photo_tip": "Use burst mode — capture mid-stott leap for iconic shot.",
        "big_five": False,
    },
    "grant's gazelle": {
        "swahili": "Swala Granti",
        "habitat": "Open savanna, semi-arid grassland",
        "diet": "Herbivore — grass and browse",
        "behavior": "Larger than Thomson's gazelle. More tolerant of dry conditions.",
        "photo_tip": "Longer horns than Thomson's — great side-profile shot.",
        "big_five": False,
    },
    "grant gazelle": {
        "swahili": "Swala Granti",
        "habitat": "Open savanna, semi-arid grassland",
        "diet": "Herbivore — grass and browse",
        "behavior": "Larger than Thomson's gazelle. More tolerant of dry conditions.",
        "photo_tip": "Longer horns than Thomson's — great side-profile shot.",
        "big_five": False,
    },
    "common eland": {
        "swahili": "Pofu",
        "habitat": "Open woodland, bushveld, montane areas",
        "diet": "Herbivore — grass, leaves, fruit",
        "behavior": "Largest African antelope. Both sexes have horns. Surprisingly agile jumpers.",
        "photo_tip": "Frame the spiral horns against open sky. Morning light best.",
        "big_five": False,
    },
    "colobus monkey": {
        "swahili": "Mbega",
        "habitat": "Dense forest canopy",
        "diet": "Herbivore — leaves, seeds, fruit",
        "behavior": "Striking black-and-white coat. Lives in forest canopy. Loud calls.",
        "photo_tip": "Look up in forest canopy. Use high ISO in low forest light.",
        "big_five": False,
    },
    "secretary bird": {
        "swahili": "Ndege wa Katibu",
        "habitat": "Open grassland, savanna",
        "diet": "Carnivore — snakes, lizards, rodents",
        "behavior": "Hunts on foot, stomps prey to death. One of Africa's most elegant birds.",
        "photo_tip": "Low angle ground shot shows the striking legs and crest feathers.",
        "big_five": False,
    },
    "somali ostrich": {
        "swahili": "Mbuni wa Kaskazini",
        "habitat": "Semi-arid savanna, bush, open plains",
        "diet": "Omnivore — seeds, plants, insects",
        "behavior": "Males have blue neck (vs pink in common ostrich). Fastest running bird.",
        "photo_tip": "Males show vivid blue neck skin — best in breeding season.",
        "big_five": False,
    },
    "african wild dog": {
        "swahili": "Mbwa Mwitu",
        "habitat": "Open savanna, woodland",
        "diet": "Carnivore — impala, wildebeest, gazelle",
        "behavior": "Most successful African predator (~80% hunt success). Critically endangered.",
        "photo_tip": "Extremely rare — photograph immediately. Report to rangers.",
        "big_five": False,
    },
})

# ─── Safari park presets with common animals ──────────────────────────────────
SAFARI_PARKS = {
    "serengeti": {
        "country": "Tanzania",
        "lat": -2.3333,
        "lng": 34.8333,
        "common_animals": ["lion", "leopard", "cheetah", "elephant", "buffalo", "wildebeest", "zebra", "giraffe", "hippopotamus", "crocodile"],
        "rare_animals": ["rhinoceros", "wild dog"],
    },
    "maasai_mara": {
        "country": "Kenya",
        "lat": -1.5000,
        "lng": 35.1500,
        "common_animals": ["lion", "leopard", "cheetah", "elephant", "buffalo", "wildebeest", "zebra", "giraffe", "hippopotamus"],
        "rare_animals": ["rhinoceros", "wild dog"],
    },
    "kruger": {
        "country": "South Africa",
        "lat": -23.9884,
        "lng": 31.5547,
        "common_animals": ["lion", "leopard", "cheetah", "elephant", "buffalo", "rhinoceros", "giraffe", "zebra", "hippopotamus", "crocodile"],
        "rare_animals": ["wild dog", "pangolin"],
    },
    "okavango": {
        "country": "Botswana",
        "lat": -19.3000,
        "lng": 22.9000,
        "common_animals": ["lion", "leopard", "cheetah", "elephant", "buffalo", "hippopotamus", "crocodile", "zebra", "giraffe"],
        "rare_animals": ["rhinoceros", "wild dog"],
    },
    "amboseli": {
        "country": "Kenya",
        "lat": -2.6527,
        "lng": 37.2606,
        "common_animals": ["elephant", "lion", "cheetah", "buffalo", "zebra", "giraffe", "hippopotamus"],
        "rare_animals": ["leopard", "rhinoceros"],
    },
    "ngorongoro": {
        "country": "Tanzania",
        "lat": -3.1667,
        "lng": 35.5833,
        "common_animals": ["lion", "leopard", "elephant", "buffalo", "rhinoceros", "zebra", "wildebeest", "hippopotamus"],
        "rare_animals": ["cheetah"],
    },
    "chobe": {
        "country": "Botswana",
        "lat": -17.8000,
        "lng": 24.8000,
        "common_animals": ["elephant", "buffalo", "lion", "leopard", "hippopotamus", "crocodile", "zebra"],
        "rare_animals": ["cheetah", "wild dog"],
    },
    "hwange": {
        "country": "Zimbabwe",
        "lat": -18.6300,
        "lng": 26.4900,
        "common_animals": ["elephant", "lion", "leopard", "buffalo", "giraffe", "zebra"],
        "rare_animals": ["cheetah", "rhinoceros", "wild dog"],
    },
}

# ─── Regional wildlife by area ────────────────────────────────────────────────
REGIONAL_WILDLIFE = {
    "east_africa": ["lion", "leopard", "cheetah", "elephant", "buffalo", "rhinoceros", "giraffe", "zebra", "wildebeest", "hippopotamus", "crocodile", "hyena"],
    "southern_africa": ["lion", "leopard", "cheetah", "elephant", "buffalo", "rhinoceros", "giraffe", "zebra", "hippopotamus", "crocodile", "wild dog"],
    "west_africa": ["lion", "leopard", "elephant", "buffalo", "hippopotamus", "crocodile", "chimpanzee", "gorilla"],
    "central_africa": ["gorilla", "chimpanzee", "elephant", "buffalo", "leopard", "hippopotamus", "crocodile"],
}

# ─── Vehicle safety rules ─────────────────────────────────────────────────────
VEHICLE_RULES = [
    "🚗 Remain in your vehicle at all times",
    "🔇 Switch off engine and minimize noise",
    "📸 Keep arms and heads inside the vehicle",
    "🚫 Do not feed or approach wildlife",
    "⏱ Observe the 5-minute rule — don't monopolize a sighting",
]


class WildlifeAgent:
    """
    Safari-optimized agentic layer.
    Provides Big Five tracking, guide-quality summaries, and tourist-friendly recommendations.
    Now includes location-based filtering for regional wildlife.
    """

    def __init__(self, detector: WildlifeDetector):
        self.detector = detector
        self.memory: dict[str, deque] = defaultdict(lambda: deque(maxlen=MEMORY_WINDOW))
        self.frame_count: dict[str, int] = defaultdict(int)
        self.last_detection_time: dict[str, float] = defaultdict(float)
        # Big Five tracker per session
        self.big_five_seen: set = set()
        self.sighting_count: dict[str, int] = defaultdict(int)
        # Location context
        self.current_park: Optional[str] = None
        self.current_region: Optional[str] = None
    
    def _detect_park_from_location(self, location: str) -> Optional[str]:
        """Detect which safari park based on location string."""
        location_lower = location.lower()
        for park_name in SAFARI_PARKS.keys():
            if park_name.replace("_", " ") in location_lower or park_name in location_lower:
                return park_name
        return None
    
    def _get_regional_animals(self, location: str) -> set[str]:
        """Get expected animals for a location/region."""
        # Try to match specific park first
        park = self._detect_park_from_location(location)
        if park and park in SAFARI_PARKS:
            park_data = SAFARI_PARKS[park]
            return set(park_data.get("common_animals", []) + park_data.get("rare_animals", []))
        
        # Fall back to regional matching
        location_lower = location.lower()
        if any(term in location_lower for term in ["kenya", "tanzania", "uganda", "rwanda"]):
            return set(REGIONAL_WILDLIFE["east_africa"])
        elif any(term in location_lower for term in ["south africa", "botswana", "zimbabwe", "namibia", "zambia"]):
            return set(REGIONAL_WILDLIFE["southern_africa"])
        elif any(term in location_lower for term in ["senegal", "ghana", "nigeria", "cameroon"]):
            return set(REGIONAL_WILDLIFE["west_africa"])
        elif any(term in location_lower for term in ["congo", "gabon", "central african"]):
            return set(REGIONAL_WILDLIFE["central_africa"])
        
        # Return all animals if location unknown
        return set(ANIMAL_INFO.keys())
    
    def _filter_detections_by_location(self, detections: list[dict], location: str) -> tuple[list[dict], list[dict]]:
        """Filter detections into expected and unexpected animals for the location."""
        expected_animals = self._get_regional_animals(location)
        
        expected = []
        unexpected = []
        
        for det in detections:
            label = det["label"].lower()
            if label in expected_animals:
                expected.append(det)
            else:
                unexpected.append(det)
        
        return expected, unexpected
    
    def _get_park_info(self, location: str) -> Optional[dict]:
        """Get park information if location matches a known park."""
        park = self._detect_park_from_location(location)
        if park and park in SAFARI_PARKS:
            return SAFARI_PARKS[park]
        return None

    async def analyze(self, frame: np.ndarray, location: str, camera_id: str) -> dict:
        self.frame_count[camera_id] += 1

        loop = asyncio.get_event_loop()
        detections = await loop.run_in_executor(None, self.detector.detect, frame)
        
        # Filter detections by location
        expected, unexpected = self._filter_detections_by_location(detections, location)
        
        # Update park context
        self.current_park = self._detect_park_from_location(location)

        now = time.time()
        if detections:
            self.last_detection_time[camera_id] = now
            for det in detections:
                label = det["label"].lower()
                self.memory[camera_id].append({
                    "label": label,
                    "confidence": det["confidence"],
                    "priority": det["priority"],
                    "timestamp": datetime.utcnow().isoformat(),
                })
                self.sighting_count[label] += 1
                if label in BIG_FIVE:
                    self.big_five_seen.add(label)

        summary = self._generate_summary(detections, location, camera_id)
        recommendations = self._generate_recommendations(detections, camera_id, location)
        behavior = self._infer_behavior(detections, camera_id)
        danger_score = self._compute_danger_score(detections)
        animal_facts = self._get_animal_facts(detections)
        big_five_status = self._big_five_status(detections)
        
        # Get location context
        park_info = self._get_park_info(location)
        expected_animals = list(self._get_regional_animals(location))

        return {
            "camera_id": camera_id,
            "location": location,
            "park": self.current_park,
            "park_info": park_info,
            "expected_animals": expected_animals,
            "frame_number": self.frame_count[camera_id],
            "animals": detections,
            "expected_detections": expected,
            "unexpected_detections": unexpected,
            "animal_count": len(detections),
            "detected": len(detections) > 0,
            "danger_score": danger_score,
            "behavior": behavior,
            "summary": summary,
            "recommendations": recommendations,
            "animal_facts": animal_facts,
            "big_five_status": big_five_status,
            "big_five_seen": list(self.big_five_seen),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

    def _generate_summary(self, detections: list, location: str, camera_id: str) -> str:
        if not detections:
            recent = list(self.memory[camera_id])
            if recent:
                last = recent[-1]
                label = last["label"].title()
                swahili = ANIMAL_INFO.get(last["label"], {}).get("swahili", "")
                swahili_str = f" ({swahili})" if swahili else ""
                return (
                    f"No wildlife currently visible at {location}. "
                    f"Last sighting: {label}{swahili_str} — {last['timestamp'][11:16]} UTC. "
                    f"Continue scanning — animals may be resting in shade."
                )
            return (
                f"Area clear at {location}. Keep scanning the treeline and water sources. "
                f"Early morning and late afternoon offer the best sighting opportunities."
            )

        labels = [d["label"] for d in detections]
        counts: dict[str, int] = {}
        for l in labels:
            counts[l] = counts.get(l, 0) + 1

        parts = []
        for animal, count in counts.items():
            info = ANIMAL_INFO.get(animal, {})
            swahili = info.get("swahili", "")
            name = f"{animal.title()}" + (f" ({swahili})" if swahili else "")
            parts.append(f"{count} {name}" if count > 1 else name)

        animal_str = ", ".join(parts)

        max_priority = max(
            (ALERT_PRIORITY.get(d["label"], "low") for d in detections),
            key=lambda p: {"low": 0, "medium": 1, "high": 2, "critical": 3}[p],
        )

        # Check Big Five
        big_five_detected = [d["label"] for d in detections if d["label"].lower() in BIG_FIVE]
        big_five_str = ""
        if big_five_detected:
            total_seen = len(self.big_five_seen)
            big_five_str = f" 🏆 Big Five sighting! ({total_seen}/5 seen this session)"

        tone = {
            "critical": f"⚠️ DANGER — Stay in vehicle! {animal_str} detected at {location}.{big_five_str}",
            "high":     f"🔴 High-priority wildlife! {animal_str} spotted at {location}.{big_five_str}",
            "medium":   f"🟡 Wildlife activity! {animal_str} at {location}.{big_five_str}",
            "low":      f"🟢 Wildlife spotted! {animal_str} at {location}.{big_five_str}",
        }[max_priority]

        # Add behavior context
        behavior = self._infer_behavior(detections, camera_id)
        behavior_notes = {
            "group_movement": " Herd on the move — excellent photo opportunity!",
            "stationary_or_foraging": " Animal appears to be feeding — great for observation.",
            "solitary_movement": " Solitary individual — possibly hunting or patrolling territory.",
        }
        tone += behavior_notes.get(behavior, "")

        return tone

    def _generate_recommendations(self, detections: list, camera_id: str, location: str = "Unknown") -> list[str]:
        if not detections:
            # Add location-specific recommendations
            park_info = self._get_park_info(location)
            base_recs = [
                "🔭 Scan water sources, shade trees, and open plains",
                "🌅 Golden hour (6–8am, 4–6pm) offers best sighting chances",
                "🤫 Keep voices low — sound travels far on the savanna",
            ]
            
            if park_info:
                common = park_info.get("common_animals", [])
                if common:
                    animals_str = ", ".join([a.title() for a in common[:5]])
                    base_recs.append(f"📍 Common in {location}: {animals_str}")
            
            return base_recs

        recs = []
        priorities = {d["priority"] for d in detections}
        labels = {d["label"].lower() for d in detections}
        
        # Check if detected animals are expected in this location
        expected_animals = self._get_regional_animals(location)
        unexpected_animals = [l for l in labels if l not in expected_animals]
        
        if unexpected_animals:
            unexpected_str = ", ".join([a.title() for a in unexpected_animals])
            recs.append(f"⚠️ Unusual sighting for {location}: {unexpected_str} — verify and report!")

        # Always include vehicle safety for any sighting
        recs.append(VEHICLE_RULES[0])  # Stay in vehicle

        if "critical" in priorities:
            recs.append("🚨 Alert your ranger/guide immediately via radio")
            recs.append("🚗 Do NOT exit the vehicle under any circumstances")
            recs.append("📸 Photograph from inside vehicle — keep windows partially open")
            recs.append("🔇 Switch off engine to reduce disturbance")

        if "high" in priorities:
            recs.append("⚠️ Maintain minimum safe distance — do not approach")
            recs.append("🔇 Minimize noise and sudden movements")
            recs.append("📻 Notify other vehicles in the area via radio")

        # Species-specific safari advice
        if any(l in labels for l in ["lion", "cheetah", "leopard"]):
            recs.append("🦁 Big cat sighting! Keep engine running for quick exit if needed")
            recs.append("📸 Use burst mode — predators move fast")

        if "leopard" in labels:
            recs.append("🌳 Check nearby trees — leopards often rest in branches with kills")

        if "cheetah" in labels:
            recs.append("🏃 Cheetah may be preparing to hunt — watch for prey animals nearby")

        if any(l in labels for l in ["elephant", "rhinoceros", "hippopotamus"]):
            recs.append("🐘 Large animal — give extra space, especially with calves present")
            recs.append("🚗 Position vehicle sideways for best photos and quick retreat")

        if "elephant" in labels:
            recs.append("👂 Watch for ear flapping and trunk raising — signs of agitation")

        if "rhinoceros" in labels:
            recs.append("🦏 Critically endangered — report exact GPS location to ranger station")
            recs.append("📍 This is a rare sighting — log it in the park registry")

        if "buffalo" in labels:
            recs.append("🐃 Buffalo are unpredictable — especially lone bulls ('dagga boys')")

        if any(l in labels for l in ["crocodile", "hippopotamus"]):
            recs.append("💧 Stay well away from water's edge — never approach on foot")

        if "giraffe" in labels:
            recs.append("📸 Silhouette against sky makes stunning photos — try wide angle")

        if "wildebeest" in labels or "zebra" in labels:
            recs.append("🦁 Large prey herds attract predators — scan surroundings carefully")

        # Photography tips
        animal_list = list(labels)
        if animal_list:
            first = animal_list[0]
            tip = ANIMAL_INFO.get(first, {}).get("photo_tip", "")
            if tip:
                recs.append(f"📸 Photo tip: {tip}")

        # Big Five bonus
        big_five_detected = [l for l in labels if l in BIG_FIVE]
        if big_five_detected:
            total = len(self.big_five_seen)
            remaining = BIG_FIVE - self.big_five_seen
            if remaining:
                remaining_str = ", ".join(a.title() for a in remaining)
                recs.append(f"🏆 Big Five progress: {total}/5 seen. Still looking for: {remaining_str}")
            else:
                recs.append("🏆 CONGRATULATIONS! You've completed the Big Five! 🎉")

        # Repeat sighting
        recent_labels = [m["label"] for m in self.memory[camera_id]]
        for label in labels:
            count = recent_labels.count(label)
            if count >= 3:
                recs.append(f"📍 {label.title()} has been in this area {count} times — likely territory or feeding ground")

        return recs

    def _get_animal_facts(self, detections: list) -> list[dict]:
        """Return safari fact cards for detected animals."""
        facts = []
        seen = set()
        for d in detections:
            label = d["label"].lower()
            if label in seen:
                continue
            seen.add(label)
            info = ANIMAL_INFO.get(label)
            if info:
                facts.append({
                    "label": label,
                    "swahili": info.get("swahili", ""),
                    "habitat": info.get("habitat", ""),
                    "diet": info.get("diet", ""),
                    "behavior": info.get("behavior", ""),
                    "photo_tip": info.get("photo_tip", ""),
                    "big_five": info.get("big_five", False),
                })
        return facts

    def _big_five_status(self, detections: list) -> dict:
        """Return Big Five completion status."""
        detected_now = {d["label"].lower() for d in detections if d["label"].lower() in BIG_FIVE}
        return {
            "total_seen": len(self.big_five_seen),
            "seen": list(self.big_five_seen),
            "remaining": list(BIG_FIVE - self.big_five_seen),
            "detected_now": list(detected_now),
            "complete": len(self.big_five_seen) == 5,
        }

    def _infer_behavior(self, detections: list, camera_id: str) -> str:
        if not detections:
            return "none"

        recent = list(self.memory[camera_id])
        recent_labels = [m["label"] for m in recent[-5:]] if recent else []
        current_labels = [d["label"] for d in detections]

        if len(detections) > 3:
            return "group_movement"
        if any(l in recent_labels for l in current_labels):
            return "stationary_or_foraging"
        if len(detections) == 1:
            return "solitary_movement"
        return "unknown"

    def _compute_danger_score(self, detections: list) -> float:
        if not detections:
            return 0.0
        priority_scores = {"critical": 1.0, "high": 0.75, "medium": 0.45, "low": 0.15}
        scores = [
            priority_scores.get(d["priority"], 0.1) * d["confidence"]
            for d in detections
        ]
        return round(min(1.0, max(scores)), 3)

# Made with Bob
