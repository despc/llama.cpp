import pytest
from utils import *

server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.model_alias = "tinyllama-2"
    server.model_tags = "tiny,story"


def test_unknown_model_is_served_by_default():
    global server
    server.start()
    res = server.make_request("POST", "/chat/completions", data={
        "model": "some-model-we-never-loaded",
        "max_tokens": 8,
        "messages": [{"role": "user", "content": "Book"}],
    })
    assert res.status_code == 200


@pytest.mark.parametrize("model_name", [
    "tinyllama-2",  # the model's own name
    "",             # "whatever you have loaded"
])
def test_check_model_name_accepts_own_names(model_name):
    global server
    server.check_model_name = True
    server.start()
    res = server.make_request("POST", "/chat/completions", data={
        "model": model_name,
        "max_tokens": 8,
        "messages": [{"role": "user", "content": "Book"}],
    })
    assert res.status_code == 200


def test_check_model_name_accepts_omitted_model():
    global server
    server.check_model_name = True
    server.start()
    res = server.make_request("POST", "/chat/completions", data={
        "max_tokens": 8,
        "messages": [{"role": "user", "content": "Book"}],
    })
    assert res.status_code == 200


@pytest.mark.parametrize("model_name", [
    "some-model-we-never-loaded",
    "TINYLLAMA-2",  # names are matched exactly, as OAI does
    "tiny",         # a tag is not a name: --tags is informational
])
def test_check_model_name_rejects_unknown_model(model_name):
    global server
    server.check_model_name = True
    server.start()
    res = server.make_request("POST", "/chat/completions", data={
        "model": model_name,
        "max_tokens": 8,
        "messages": [{"role": "user", "content": "Book"}],
    })
    assert res.status_code == 404
    assert res.body["error"]["type"] == "not_found_error"
    # the generic routing 404 must not have swallowed our message
    assert model_name in res.body["error"]["message"]


@pytest.mark.parametrize("endpoint,data", [
    ("/completions",     {"prompt": "Book", "n_predict": 4}),
    ("/v1/completions",  {"prompt": "Book", "max_tokens": 4}),
    ("/v1/responses",    {"input": "Book"}),
    ("/v1/messages",     {"max_tokens": 4, "messages": [{"role": "user", "content": "Book"}]}),
    ("/apply-template",  {"messages": [{"role": "user", "content": "Book"}]}),
])
def test_check_model_name_covers_the_other_endpoints(endpoint, data):
    global server
    server.check_model_name = True
    server.start()
    res = server.make_request("POST", endpoint, data={**data, "model": "some-model-we-never-loaded"})
    assert res.status_code == 404


def test_check_model_name_leaves_unknown_routes_alone():
    global server
    server.check_model_name = True
    server.start()
    res = server.make_request("GET", "/no/such/route")
    assert res.status_code == 404
    assert res.body["error"]["message"] == "File Not Found"
