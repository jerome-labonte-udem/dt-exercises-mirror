duckietown_protocols package
============================

.. contents::

The ``duckietown_protocols`` package contains all the protocols that are needed for multi-robot interaction. For example,
they match a semantic meaning to the different patterns of colors and blinking of LEDs. For the multi-robot system to
work is important that every robot and every container inside each robot uses the same set of protocols (i.e. same
version of ``dt-ros-commons``.


Protocols
---------

LED protocol
^^^^^^^^^^^^

It defines three main points: the intensity to be used by LEDs, the channel ordering (``channel_order``) of each type of robot (using LEDs)
and the actual ``LED_protocol``. The latter has a dict of colors defined as ``[R,G,B]`` triplets, where each entry ranges from
0 to 1. This triplets are then reordered using the ``channel_order`` dependant on the robot type. The ``LED_protocol`` also
specifies a set of fixed frequencies (`Nyquist frequency <https://en.wikipedia.org/wiki/Nyquist_frequency>`_). Afterwards a set of ``signals`` is defined, in which
each signal name has:

- ``color_mask``: specifies to which LEDs to modify the color

- ``color_list``: specifies which color should each LED have (one value means all of them)

- ``frequency_mask``: in case of blinking LEDs, specifies which ones should blink

- ``frequency``: pointer to the predefined frequencies, assigning one to each signal
