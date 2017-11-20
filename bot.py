from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging

import sys
import warnings

from rasa_core.actions import Action
from rasa_core.agent import Agent
from rasa_core.channels.rest import HttpInputChannel
from rasa_core.interpreter import RasaNLUInterpreter
from rasa_core.policies.keras_policy import KerasPolicy
from rasa_core.policies.memoization import MemoizationPolicy
from rasa_core.channels.rest import HttpInputComponent
from rasa_core.channels.channel import UserMessage, OutputChannel
from rasa_core.channels.console import ConsoleOutputChannel
from rasa_core import utils

logger = logging.getLogger(__name__)


class ChangeDeviceState(Action):
    def name(self):
        return 'action_change_device_state'

    def run(self, dispatcher, tracker, domain):
        import requests
        import json
        headers = {'content-type': 'application/json'}
        url = "http://localhost:8060/ai/command_rasa"

        data = {"state": tracker.get_slot('state'), "device": tracker.get_slot('device'),"location":tracker.get_slot('location')}
        #params = {'sessionKey': '9ebbd0b25760557393a43064a92bae539d962103','format': 'xml', 'platformId': 1}
        state = tracker.get_slot('state')
        device = tracker.get_slot('device')
        location = tracker.get_slot('location')

        if state and device and location:
            requests.post(url,data=json.dumps(data), headers=headers)
            dispatcher.utter_message("changing state of device..")
            default_output_color = utils.bcolors.OKBLUE
            utils.print_color(tracker.get_slot('state'), default_output_color)
            utils.print_color(tracker.get_slot('device'), default_output_color)
            utils.print_color(tracker.get_slot('location'), default_output_color)
        else:
            utils.print_color(tracker.get_slot('state'), default_output_color)
            utils.print_color(tracker.get_slot('device'), default_output_color)
            utils.print_color(tracker.get_slot('location'), default_output_color)
        return []


class ShowDeviceState(Action):
    def name(self):
        return 'action_show_device_state'

    def run(self, dispatcher, tracker, domain):
        dispatcher.utter_message("device state changed.")
        return []


class RestaurantPolicy(KerasPolicy):
    def model_architecture(self, num_features, num_actions, max_history_len):
        """Build a Keras model and return a compiled model."""
        from keras.layers import LSTM, Activation, Masking, Dense
        from keras.models import Sequential

        n_hidden = 32  # size of hidden layer in LSTM
        # Build Model
        batch_shape = (None, max_history_len, num_features)

        model = Sequential()
        model.add(Masking(-1, batch_input_shape=batch_shape))
        model.add(LSTM(n_hidden, batch_input_shape=batch_shape))
        model.add(Dense(input_dim=n_hidden, output_dim=num_actions))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        logger.debug(model.summary())
        return model


class WebhookOutputChannel(OutputChannel):
    """Simple bot that outputs the bots messages to the a webhook"""

    default_output_color = utils.bcolors.OKBLUE
    url = ""

    def __init__(self,url):
        self.url=url

    def send_text_message(self, recipient_id, message):
        # type: (Text, Text) -> None
        import requests
        import json
        headers = {'content-type': 'application/json'}
        #url = "http://localhost:8060/ai/message_rasa"

        data = {"speech": message, "recipient": recipient_id}
        #params = {'sessionKey': '9ebbd0b25760557393a43064a92bae539d962103','format': 'xml', 'platformId': 1}

        requests.post(self.url,data=json.dumps(data), headers=headers)
        
        #utils.print_color(message, self.default_output_color)
        #utils.print_color(recipient_id, self.default_output_color)


class PrintInputComponent(HttpInputComponent):
    def blueprint(self, on_new_message):
        from flask import Blueprint
        from flask import jsonify, request

        bot = Blueprint("bot", __name__)

        @bot.route("/", methods=['GET', 'POST'])
        def health():
            return jsonify({"status": "ok"})

        @bot.route("/chat", methods=['GET', 'POST'])
        def chat():
            output = request.get_json()
            user_msg = UserMessage(
                output["q"], WebhookOutputChannel("http://localhost:8060/ai/message_rasa"), output["recipient"])
            on_new_message(user_msg)
            return 'success'

        return bot


def train_dialogue(domain_file="home_control_domain.yml",
                   model_path="models/dialogue",
                   training_data_file="data/stories.md"):
    agent = Agent(domain_file,
                  policies=[MemoizationPolicy(), RestaurantPolicy()])

    agent.train(
        training_data_file,
        max_history=3,
        epochs=100,
        batch_size=50,
        augmentation_factor=50,
        validation_split=0.2
    )

    agent.persist(model_path)
    return agent


def train_nlu():
    from rasa_nlu.converters import load_data
    from rasa_nlu.config import RasaNLUConfig
    from rasa_nlu.model import Trainer

    training_data = load_data('data/control_data.json')
    trainer = Trainer(RasaNLUConfig("nlu_model_config.json"))
    trainer.train(training_data)
    model_directory = trainer.persist(
        'models/nlu/', fixed_model_name="current")

    return model_directory


def run(serve_forever=True):
    agent = Agent.load("models/dialogue",
                       interpreter=RasaNLUInterpreter("models/nlu/current"))

    if serve_forever:
        agent.handle_channel(HttpInputChannel(
            5001, '/app', PrintInputComponent()))
    return agent


if __name__ == '__main__':
    logging.basicConfig(level="DEBUG")

    parser = argparse.ArgumentParser(
        description='starts the bot')

    parser.add_argument(
        'task',
        choices=["train-nlu", "train-dialogue", "run"],
        help="what the bot should do - e.g. run or train?")
    task = parser.parse_args().task

    # decide what to do based on first parameter of the script
    if task == "train-nlu":
        train_nlu()
    elif task == "train-dialogue":
        train_dialogue()
    elif task == "run":
        run()
    else:
        warnings.warn("Need to pass either 'train-nlu', 'train-dialogue' or "
                      "'run' to use the script.")
        exit(1)
