{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. add toctree option to make autodoc generate the pages

.. autoclass:: {{ objname }}

{# Order the time axis (n_t) right after n_vars, mirroring the shape tuple. #}
{% if "n_t" in attributes and "n_vars" in attributes %}
{% set _ns = namespace(a=[]) %}
{% for item in attributes if item != "n_t" %}
{% set _ns.a = _ns.a + [item] %}
{% if item == "n_vars" %}{% set _ns.a = _ns.a + ["n_t"] %}{% endif %}
{% endfor %}
{% set attributes = _ns.a %}
{% endif %}

{% block attributes %}
{% if attributes %}
Attributes table
~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree:
{% for item in attributes %}
    ~{{ name }}.{{ item }}
{%- endfor %}
{% endif %}
{% endblock %}

{% block methods %}
{% if methods %}
Methods table
~~~~~~~~~~~~~

.. autosummary::
    :toctree:
{% for item in methods %}
    {%- if item != '__init__' %}
    ~{{ name }}.{{ item }}
    {%- endif -%}
{%- endfor %}
{% endif %}
{% endblock %}
