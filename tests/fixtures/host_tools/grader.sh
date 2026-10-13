call() {
    curl -fsS -X POST -H 'Content-Type: application/json' -d "$1" http://host.shellsim/tools
}
conversation=$(call '{"tool":"conversation.list","arguments":{}}' | jq -r '.result.messages[0]')
answer=$(call '{"tool":"workspace.read_file","arguments":{"path":"/work/answer.txt"}}' | jq -r .result.data_base64 | base64 -d)
score=0.0
if [ "$conversation" = 'What is 6 * 7?' ] && [ "$answer" = 42 ]; then
    score=1.0
fi
call "{\"tool\":\"grade.submit\",\"arguments\":{\"score\":$score}}" > /dev/null
printf '%s\n' "$score"
