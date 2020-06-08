import rospy
from rospy.impl.tcpros import DEFAULT_BUFF_SIZE
from rospy.impl.registration import get_topic_manager

class DTSubscriber(rospy.Subscriber):
    """ A wrapper around ``rospy.Subscriber``.

    This class is exactly the same as the standard
    `rospy.Subscriber <http://docs.ros.org/api/rospy/html/rospy.topics.Subscriber-class.html>`_
    with the only difference of an :py:attr:`active` attribute being added. Whenever the :py:meth:`publish` method is used,
    an actual message will be send only if :py:attr:`active` is set to ``True``.

    Args:
       name (:obj:`str`): resource name of topic, e.g. 'laser'
       data_class (:obj:`ROS Message class`): message class for serialization
       subscriber_listener (:obj:`SubscribeListener`): listener for subscription events. May be `None`
       tcp_nodelay (:obj:`bool`): If ``True``, sets ``TCP_NODELAY`` on publisher's socket (disables Nagle algorithm).
          This results in lower latency publishing at the cost of efficiency.
       latch (:obj:`bool`) - If ``True``, the last message published is 'latched', meaning that any future subscribers
          will be sent that message immediately upon connection.
       headers (:obj:`dict`) - If not ``None``, a dictionary with additional header key-values being
          used for future connections.
       queue_size (:obj:`int`) - The queue size used for asynchronously publishing messages from different
          threads. A size of zero means an infinite queue, which can be dangerous. When the keyword is not
          being used or when ``None`` is passed all publishing will happen synchronously and a warning message
          will be printed.

    Attributes:
       All standard rospy.Publisher attributes
       active (:obj:`bool`): A flag that if set to ``True`` will allow publishing. If set to ``False``, any calls
          to `publish` will not result in a message being sent. Can be directly assigned.

    Raises:
       ROSException: if parameters are invalid

    """

    def __init__(self, name, data_class, callback=None, callback_args=None, queue_size=None,
                 buff_size=DEFAULT_BUFF_SIZE, tcp_nodelay=False):

        super(DTSubscriber, self).__init__(name, data_class, callback=callback, callback_args=callback_args,
                                           queue_size=queue_size, buff_size=buff_size, tcp_nodelay=tcp_nodelay)

        self._active = True
        self._attributres_keeper = {'name': name,
                                     'data_class': data_class,
                                     'callback': callback,
                                     'callback_args': callback_args,
                                     'queue_size': queue_size,
                                     'buff_size': buff_size,
                                     'tcp_nodelay': tcp_nodelay,
                                     'resolved_name': self.resolved_name}


    @property  #: Read-only property for the private attributes
    def active(self):
        return self._active

    @active.setter  #: Setter for the read-only property for the private attributes
    def active(self, new_status):
        if self._active==new_status:
            # Don't do anything if the status doesn't change
            pass
        elif self._active==False and new_status==True:
            # Reactive the subscription
            self.reregister()
        else:
            # Unsubscribe
            self.unregister()

        self._active = new_status

    def reregister(self):
        """
        Resets some of the attributes in order to restore the state of the object before
        the ``unregister`` method was called

        Reversing the ``Topic.unregister()`` and ``Subscriber.unregister()`` methods from
        `here <http://docs.ros.org/api/rospy/html/rospy.topics-pysrc.html>`_.

        """

        # Restore what was removed by the Topic.unregister() method
        self.impl = get_topic_manager().acquire_impl(self.reg_type, self._attributres_keeper['resolved_name'],
                                                     self._attributres_keeper['data_class'])
        self.resolved_name = self._attributres_keeper['resolved_name']
        self.data_class = self._attributres_keeper['data_class']
        self.type = self.data_class._type
        self.md5sum = self.data_class._md5sum

        # Restore what was removed by the Subscriber.unregister() method
        if self._attributres_keeper['queue_size'] is not None:
            self.impl.set_queue_size(self._attributres_keeper['queue_size'])
        if self._attributres_keeper['buff_size'] != DEFAULT_BUFF_SIZE:
            self.impl.set_buff_size(self._attributres_keeper['buff_size'])
        if self._attributres_keeper['callback'] is not None:
            self.impl.add_callback(self._attributres_keeper['callback'], self._attributres_keeper['callback_args'])
            self.callback = self._attributres_keeper['callback']
            self.callback_args = self._attributres_keeper['callback_args']
        else:
            self.callback = self.callback_args = None
        if self._attributres_keeper['tcp_nodelay']:
            self.impl.set_tcp_nodelay(self._attributres_keeper['tcp_nodelay'])