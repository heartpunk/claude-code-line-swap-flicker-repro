from hypothesis import HealthCheck, settings

# suppress differing_executors: mutmut's trampoline mechanism changes the
# execution context, causing Hypothesis to flag tests run under mutation
# testing. This is expected behaviour and not a test correctness issue.
settings.register_profile(
    "mutmut",
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.differing_executors],
)
settings.register_profile(
    "default",
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile("default")
