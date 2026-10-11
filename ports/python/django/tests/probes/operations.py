"""Exercise local public behavior in the actual guest."""

from django import forms
from django.conf import settings
from django.http import QueryDict
from django.urls import path, resolve

settings.configure(DEFAULT_CHARSET="utf-8", USE_I18N=False, SECRET_KEY="guest-test")


class CountForm(forms.Form):
    count = forms.IntegerField(min_value=0)


assert CountForm({"count": "7"}).is_valid()
assert not CountForm({"count": "-1"}).is_valid()
assert QueryDict("x=1&x=2").getlist("x") == ["1", "2"]


def view(request, value):
    return value


assert resolve("/item/7/", urlconf=(path("item/<int:value>/", view),)).kwargs == {"value": 7}
print("django: guest probe passed")
