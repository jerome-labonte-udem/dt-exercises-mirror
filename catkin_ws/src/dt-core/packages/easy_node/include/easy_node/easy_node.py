from UserDict import UserDict
from contextlib import contextmanager
import rospy
import threading

import duckietown_utils as dtu

from .node_description.configuration import PROCESS_THREADED, PROCESS_SYNCHRONOUS
from .node_description.configuration import load_configuration_package_node
from .user_config.decide import get_user_configuration
from .utils.timing import ProcessingTimingStats


__all__ = [
    'EasyNode',
]


class EasyNode(object):

    ENV = dtu.DuckietownConstants.DUCKIETOWN_CONFIG_SEQUENCE_variable

    def __init__(self, package_name, node_type_name):
        self.package_name = package_name
        self.node_type_name = node_type_name
        rospy.init_node(node_type_name, anonymous=False)  # @UndefinedVariable

    def _msg(self, msg):
        return '%s | %s' % (self.node_type_name, msg)

    def info(self, msg):
        msg = self._msg(msg)
        rospy.loginfo(msg)  # @UndefinedVariable

    def debug(self, msg):
        msg = self._msg(msg)
        rospy.logdebug(msg)  # @UndefinedVariable

    def error(self, msg):
        msg = self._msg(msg)
        rospy.logerror(msg)  # @UndefinedVariable

    def on_init(self):
        self.info('on_init (default)')

    def on_parameters_changed(self, first_time, changed):
        self.info('(default) First: %s Parameters changed: %s' %
                  (first_time, changed))

    def on_shutdown(self):
        self.info('on_shutdown (default)')

    def _init(self):
        c = load_configuration_package_node(
            self.package_name, self.node_type_name)
        self._configuration = c
        self._init_publishers()
        self._init_parameters()
        self._init_subscriptions()
        self.info(self._configuration)

    def _init_subscriptions(self):
        subscriptions = self._configuration.subscriptions

        class Subscribers():
            pass
        self.subscribers = Subscribers()

        class SubscriberProxy():
            def __init__(self, sub):
                self.sub = sub
                self.pts = ProcessingTimingStats()

            def init_threaded(self):
                self.thread_lock = threading.Lock()

        class Callback():
            def __init__(self, node, subscription):
                self.node = node
                self.subscription = subscription

            def __call__(self, data):
                subscriber_proxy = getattr(
                    self.node.subscribers, self.subscription.name)
                self.node._sub_callback(
                    self.subscription, subscriber_proxy, data)

        for s in subscriptions.values():
            callback = Callback(node=self, subscription=s)
            S = rospy.Subscriber(s.topic, s.type, callback, # @UndefinedVariable
                                 queue_size=s.queue_size)  # @UndefinedVariable
            sp = SubscriberProxy(S)
            setattr(self.subscribers, s.name, sp)

            self.info('Subscribed to %s' % s.topic)
            if s.process == PROCESS_THREADED:
                sp.init_threaded()

    def _sub_callback(self, subscription, subscriber_proxy, data):
        subscriber_proxy.pts.received_message(data)
        callback_name = 'on_received_%s' % subscription.name
        if hasattr(self, callback_name):
            if subscription.process == PROCESS_SYNCHRONOUS:
                # Call directly
                subscriber_proxy.pts.decided_to_process(data)
                self._call_callback(callback_name, subscription, data)
            elif subscription.process == PROCESS_THREADED:
                # Start a daemon thread to process the image
                target = self._sub_callback_threaded
                args = (callback_name, subscription, subscriber_proxy, data)
                thread = threading.Thread(target=target, args=args)
                thread.setDaemon(True)
                thread.start()
            else:
                assert False, subscription.process
        else:
            subscriber_proxy.pts.decided_to_skip()
            self.info('No callback %r defined.' % callback_name)

    def _get_context(self, subscription):
        class Context():
            def __init__(self, node, subscription):
                self.node = node
                self.subscription = subscription
                self.sp = getattr(node.subscribers, subscription.name)

            @contextmanager
            def phase(self, name):
                with self.sp.pts.phase(name):
                    yield

            def get_stats(self):
                return self.sp.pts.get_stats()

        context = Context(self, subscription)
        return context

    def _sub_callback_threaded(self, callback_name, subscription, subscriber_proxy, data):
        if not subscriber_proxy.thread_lock.acquire(False):
            # TODO self.stats.skipped()
            subscriber_proxy.pts.decided_to_skip()
            return
        try:
            subscriber_proxy.pts.decided_to_process(data)
            self._call_callback(callback_name, subscription, data)
        finally:
            # Release the thread lock
            subscriber_proxy.thread_lock.release()

    def _call_callback(self, callback_name, subscription, data):
        c = getattr(self, callback_name)
        context = self._get_context(subscription)
        try:
            c(context, data)
        finally:
            pass

    def _init_publishers(self):
        publishers = self._configuration.publishers

        class Publishers():
            pass
        self.publishers = Publishers()
        for s in publishers.values():
            # @UndefinedVariable
            P = rospy.Publisher(
                s.topic, s.type, queue_size=s.queue_size, latch=s.latch)
            setattr(self.publishers, s.name, P)

    def _init_parameters(self):
        parameters = self._configuration.parameters
        
        class Config():
            def __getattr__(self, name):
                if not name in parameters:
                    msg = 'The user is trying to use %r, which is not a parameter ' % name
                    msg += 'for this node.\n'
                    msg += 'The declared parameters that can be used are:\n- %s' % "\n- ".join(list(parameters))
                    raise AttributeError(msg)
                return object.__getattr__(self, name)
            
        self.config = Config()
        values = {}

        # load the configuration
        self.info('Loading parameters...')
        with dtu.rospy_timeit_wall('getting configuration files'):
            qr = get_user_configuration(self.package_name, self.node_type_name)
            
        if not qr.is_complete():
            msg = '\nThe configuration that I could load is not complete:\n'
            msg += dtu.indent(str(qr), '   | ')
            msg =  dtu.indent(msg, '%s / %s fatal error >  ' %
                         (self.package_name, self.node_type_name))
            raise  dtu.DTConfigException(msg)
        self.info('Loaded configuration:\n%s' % qr)

        for p in parameters.values():
            try:
                val = qr.values.get(p.name)  # @UndefinedVariable
            except KeyError:
                msg = 'Could not load required parameter %r.' % p.name
                raise  dtu.DTConfigException(msg)

            # write to parameter server, for transparency
            if val is not None:  # we cannot set None parameters
                rospy.set_param('~' + p.name, val)  # @UndefinedVariable

            setattr(self.config, p.name, val)
            values[p.name] = val
            
        self. _on_parameters_changed(first_time=True, values=values)

        duration = self.config.en_update_params_interval
        duration = rospy.Duration.from_sec(duration)  # @UndefinedVariable
        rospy.Timer(duration, self._update_parameters)  # @UndefinedVariable

    def _on_parameters_changed(self, first_time, values):
        try:
            values1 = UpdatedParameters(**values)
            values1.set_allowed(list(self._configuration.parameters))
            self.on_parameters_changed(first_time, values1)
        except  dtu.DTConfigException as e:
            msg = 'Configuration error raised by on_parameters_changed()' 
            msg += '\n\n' +  dtu.indent( dtu.yaml_dump(values), '  ', 'Configuration: ')
            dtu.raise_wrapped( dtu.DTConfigException, e, msg, compact=True)
        except Exception as e:
            msg = 'Configuration error raised by on_parameters_changed().'
            msg += '\n\n' +  dtu.indent( dtu.yaml_dump(values), '  ', 'Configuration: ')
            dtu.raise_wrapped( dtu.DTConfigException, e, msg)
             
        
    def _update_parameters(self, _event):
        changed = self._get_changed_parameters()
#         self.info('Parameters changed: %s' % sorted(changed))
        if changed:
            for k, v in changed.items():
                setattr(self.config, k, v)
            self._on_parameters_changed(False, changed)
        else:
            pass
            # self.info('No change in parameters.')

    def _get_changed_parameters(self):
        parameters = self._configuration.parameters
        changed = {}
        for p in parameters.values():
            val = rospy.get_param('~' + p.name)  # @UndefinedVariable
            current = getattr(self.config, p.name)
            s1 = current.__repr__()
            s2 = val.__repr__()
            if s1 != s2:
                changed[p.name] = val
#                 setattr(self.config, p.name, current)
        return changed

    def spin(self):
        rospy.on_shutdown(self.on_shutdown)  # @UndefinedVariable
        self._init()
        self.on_init()
        rospy.spin()  # @UndefinedVariable


class UpdatedParameters(UserDict):
    def __init__(self, *args, **kwargs):
        UserDict.__init__(self, *args, **kwargs)
        self.allowed = None
    
    def set_allowed(self, allowed):
        self.allowed = allowed
        
    def has_key(self, x):
        
        return self.__contains__(self, x)
    def __contains__(self, x):
        if self.allowed is not None:
            if not x in self.allowed:
                msg = 'The user is trying to check that a parameter %r was updated, ' % x
                msg += 'however, no such parameter was declared (the declared ones were: %s).' % self.allowed
                raise ValueError(msg)
        return UserDict.__contains__(self, x)
    