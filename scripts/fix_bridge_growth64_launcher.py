"""Create a corrected launcher without changing the frozen experiment package."""
import hashlib
from pathlib import Path
import sys

project = Path(sys.argv[1])
source = project / "bridge_growth64_v1" / "fit.sh"
original = source.read_bytes()
assert hashlib.sha256(original).hexdigest() == "5031987d2bf923d0ee4c6d22a16c7a499235f341f9e6edf95044d6e954200241", "Unexpected original launcher; stop and inspect"
bad = b'\n+  --preflight-reference '
assert original.count(bad) == 1
corrected = original.replace(bad, b'\n  --preflight-reference ')
assert hashlib.sha256(corrected).hexdigest() == "0d126c075f7751f87c24d656c156a29dc32bed026034643bb970be20baee4edf"
destination = project / "bridge_growth64_fit_fixed.sh"
assert not destination.exists() or destination.read_bytes() == corrected, "Different corrected launcher already exists"
destination.write_bytes(corrected)
print("Corrected launcher:", destination)
print("SHA256:", hashlib.sha256(corrected).hexdigest())
