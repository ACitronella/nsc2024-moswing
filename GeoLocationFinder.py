from geopy.geocoders import Nominatim

geolocator = Nominatim(user_agent="nsc moswing")

def find_latlong(address:str):
    location = geolocator.geocode(address, namedetails=True)
    if location is None: return None
    return {"lat": location.latitude, "lon": location.longitude}