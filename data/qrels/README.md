# Julgamentos de relevância revisados

`pilot-proposed.jsonl` conserva o nome histórico, mas contém os julgamentos revisados pelo autor para as dez consultas atuais `q01`–`q10`: 530 pares, 53 chunks por pergunta. Cada par recebe nota explícita `0`, `1` ou `2`; ausência de linha seria um julgamento faltante, não uma nota zero. As linhas positivas incluem `reason`. O corpus está em `data/corpus/chunks.jsonl` e as necessidades em `data/queries/pilot-queries.jsonl`.

## Rubrica

- `2`: o chunk responde diretamente à pergunta ou explica suficientemente uma causa plausível dentro do contexto de uma pergunta aberta. O texto do chunk deve sustentar a explicação por si só.
- `1`: o chunk oferece uma verificação, uma etapa ou parte útil da solução, mas não resolve a pergunta principal sozinho.
- `0`: o chunk não acrescenta explicação ou ação útil para aquela necessidade; sobreposição de termos ou tema não basta.

A pergunta apresentada ao usuário define o foco. O campo `information_need` esclarece esse foco, sem adicionar uma segunda pergunta obrigatória. Em `q04`, a resposta principal é se o GitHub reenvia automaticamente; instruções de reentrega são contexto complementar. Em `q06`, uma explicação completa de um motivo plausível para uma listagem incompleta pode receber 2. Em `q10`, distinguir paginação de permissão exige informações dos dois lados.

`Recall@k` considera notas 1 e 2 relevantes; `nDCG@k` usa os graus. Veja `docs/query-id-migration.md` para relacionar IDs históricos aos atuais e `docs/ten-query-revision-comparison.md` para a revisão após duas auditorias independentes.
