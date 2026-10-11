# Shellingham

The guest probe detects its actual virtual Bash parent and rejects a missing
virtual process. The port reads Linux `PPid` fields from Shellsim's `/proc/status`
files when a Linux `/proc/stat` file is unavailable. Normal host Linux and BSD
stat parsing remain unchanged. Process queries use the virtual filesystem.
