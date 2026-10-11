"""Register and call an ordinary Python hook in the guest."""

import pluggy

spec = pluggy.HookspecMarker("probe")
implementation = pluggy.HookimplMarker("probe")


class Hooks:
    @spec
    def transform(self, value):
        """Return the transformed value."""


class Plugin:
    @implementation
    def transform(self, value):
        return value * 2


manager = pluggy.PluginManager("probe")
manager.add_hookspecs(Hooks)
plugin = Plugin()
manager.register(plugin)
assert manager.hook.transform(value=21) == [42]
manager.unregister(plugin)
assert manager.hook.transform(value=21) == []
print("pluggy: registration, dispatch and removal passed")
