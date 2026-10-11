"""Exercise local public behavior in the actual guest."""

from botocore.httpsession import URLLib3Session
from botocore.loaders import Loader
from botocore.model import ServiceModel

model = ServiceModel(Loader().load_service_model("s3", "service-2"), service_name="s3")
assert "ListBuckets" in model.operation_names
assert "Bucket" in model.operation_model("GetObject").input_shape.members
try:
    URLLib3Session()
except NotImplementedError:
    pass
else:
    raise AssertionError("unavailable HTTP transport accepted")
print("botocore: guest probe passed")
