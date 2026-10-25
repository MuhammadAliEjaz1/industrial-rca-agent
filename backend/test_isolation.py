import pprint
from models.failure_classifier import predict_failure

print("=== Test 2b: cleaner tool wear isolation (torque × tool_wear well below overstrain threshold) ===")
pprint.pprint(predict_failure(
    air_temperature=299.0, process_temperature=309.5,
    rotational_speed=1400, torque=35.0, tool_wear=225
))