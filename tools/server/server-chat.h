// Chat conversion functions for server (Responses API, Anthropic API, OAI streaming diffs)

#pragma once

#include "chat.h"
#include "server-common.h"
#include "server-http.h"

#include "json.h"

// Convert OpenAI Responses API format to OpenAI Chat Completions API format
json server_chat_convert_responses_to_chatcmpl(const json & body);

// Responses "namespace" tools (e.g. Codex's MCP servers) are offered to the model as plain functions named
// <namespace>__<name>. This maps each such flat name back to {namespace, name}, from the request's tool list.
using server_chat_tool_namespaces = std::map<std::string, std::pair<std::string, std::string>>;
server_chat_tool_namespaces server_chat_responses_tool_namespaces(const json & body);
std::string server_chat_flat_tool_name(const std::string & ns, const std::string & name);

// Restore name and namespace in every Responses function_call item of a response or stream event (in place)
void server_chat_restore_tool_namespaces(json & j, const server_chat_tool_namespaces & ns);

// Convert Anthropic Messages API format to OpenAI Chat Completions API format
json server_chat_convert_anthropic_to_oai(const json & body);

// convert OpenAI transcriptions API format to OpenAI Chat Completions API format
json convert_transcriptions_to_chatcmpl(
    const json & body,
    const common_chat_templates * tmpls,
    const std::map<std::string, uploaded_file> & in_files,
    std::vector<raw_buffer> & out_files);

json server_chat_msg_diff_to_json_oaicompat(const common_chat_msg_diff & diff);
