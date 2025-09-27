Filters
Filters define trading rules on a symbol or an exchange. Filters come in two forms: symbol filters and exchange filters.

Symbol filters
PRICE_FILTER
The PRICE_FILTER defines the price rules for a symbol. There are 3 parts:

minPrice defines the minimum price/stopPrice allowed; disabled on minPrice == 0.
maxPrice defines the maximum price/stopPrice allowed; disabled on maxPrice == 0.
tickSize defines the intervals that a price/stopPrice can be increased/decreased by; disabled on tickSize == 0.
Any of the above variables can be set to 0, which disables that rule in the price filter. In order to pass the price filter, the following must be true for price/stopPrice of the enabled rules:

price >= minPrice
price <= maxPrice
price % tickSize == 0
/exchangeInfo format:

{
  "filterType": "PRICE_FILTER",
  "minPrice": "0.00000100",
  "maxPrice": "100000.00000000",
  "tickSize": "0.00000100"
}

PERCENT_PRICE
The PERCENT_PRICE filter defines the valid range for the price based on the average of the previous trades. avgPriceMins is the number of minutes the average price is calculated over. 0 means the last price is used.

In order to pass the percent price, the following must be true for price:

price <= weightedAveragePrice * multiplierUp
price >= weightedAveragePrice * multiplierDown
/exchangeInfo format:

{
  "filterType": "PERCENT_PRICE",
  "multiplierUp": "1.3000",
  "multiplierDown": "0.7000",
  "avgPriceMins": 5
}

PERCENT_PRICE_BY_SIDE
The PERCENT_PRICE_BY_SIDE filter defines the valid range for the price based on the average of the previous trades.
avgPriceMins is the number of minutes the average price is calculated over. 0 means the last price is used.
There is a different range depending on whether the order is placed on the BUY side or the SELL side.

Buy orders will succeed on this filter if:

Order price <= weightedAveragePrice * bidMultiplierUp
Order price >= weightedAveragePrice * bidMultiplierDown
Sell orders will succeed on this filter if:

Order Price <= weightedAveragePrice * askMultiplierUp
Order Price >= weightedAveragePrice * askMultiplierDown
/exchangeInfo format:

  {
    "filterType": "PERCENT_PRICE_BY_SIDE",
    "bidMultiplierUp": "1.2",
    "bidMultiplierDown": "0.2",
    "askMultiplierUp": "5",
    "askMultiplierDown": "0.8",
    "avgPriceMins": 1
  }

LOT_SIZE
The LOT_SIZE filter defines the quantity (aka "lots" in auction terms) rules for a symbol. There are 3 parts:

minQty defines the minimum quantity/icebergQty allowed.
maxQty defines the maximum quantity/icebergQty allowed.
stepSize defines the intervals that a quantity/icebergQty can be increased/decreased by.
In order to pass the lot size, the following must be true for quantity/icebergQty:

quantity >= minQty
quantity <= maxQty
quantity % stepSize == 0
/exchangeInfo format:

{
  "filterType": "LOT_SIZE",
  "minQty": "0.00100000",
  "maxQty": "100000.00000000",
  "stepSize": "0.00100000"
}

MIN_NOTIONAL
The MIN_NOTIONAL filter defines the minimum notional value allowed for an order on a symbol. An order's notional value is the price * quantity. applyToMarket determines whether or not the MIN_NOTIONAL filter will also be applied to MARKET orders. Since MARKET orders have no price, the average price is used over the last avgPriceMins minutes. avgPriceMins is the number of minutes the average price is calculated over. 0 means the last price is used.

/exchangeInfo format:

{
  "filterType": "MIN_NOTIONAL",
  "minNotional": "0.00100000",
  "applyToMarket": true,
  "avgPriceMins": 5
}

NOTIONAL
The NOTIONAL filter defines the acceptable notional range allowed for an order on a symbol.

applyMinToMarket determines whether the minNotional will be applied to MARKET orders.
applyMaxToMarket determines whether the maxNotional will be applied to MARKET orders.

In order to pass this filter, the notional (price * quantity) has to pass the following conditions:

price * quantity <= maxNotional
price * quantity >= minNotional
For MARKET orders, the average price used over the last avgPriceMins minutes will be used for calculation.
If the avgPriceMins is 0, then the last price will be used.

/exchangeInfo format:

{
   "filterType": "NOTIONAL",
   "minNotional": "10.00000000",
   "applyMinToMarket": false,
   "maxNotional": "10000.00000000",
   "applyMaxToMarket": false,
   "avgPriceMins": 5
}

ICEBERG_PARTS
The ICEBERG_PARTS filter defines the maximum parts an iceberg order can have. The number of ICEBERG_PARTS is defined as CEIL(qty / icebergQty).

/exchangeInfo format:

{
  "filterType": "ICEBERG_PARTS",
  "limit": 10
}

MARKET_LOT_SIZE
The MARKET_LOT_SIZE filter defines the quantity (aka "lots" in auction terms) rules for MARKET orders on a symbol. There are 3 parts:

minQty defines the minimum quantity allowed.
maxQty defines the maximum quantity allowed.
stepSize defines the intervals that a quantity can be increased/decreased by.
In order to pass the market lot size, the following must be true for quantity:

quantity >= minQty
quantity <= maxQty
quantity % stepSize == 0
/exchangeInfo format:

{
  "filterType": "MARKET_LOT_SIZE",
  "minQty": "0.00100000",
  "maxQty": "100000.00000000",
  "stepSize": "0.00100000"
}

MAX_NUM_ORDERS
The MAX_NUM_ORDERS filter defines the maximum number of orders an account is allowed to have open on a symbol. Note that both "algo" orders and normal orders are counted for this filter.

/exchangeInfo format:

{
  "filterType": "MAX_NUM_ORDERS",
  "maxNumOrders": 25
}

MAX_NUM_ALGO_ORDERS
The MAX_NUM_ALGO_ORDERS filter defines the maximum number of "algo" orders an account is allowed to have open on a symbol. "Algo" orders are STOP_LOSS, STOP_LOSS_LIMIT, TAKE_PROFIT, and TAKE_PROFIT_LIMIT orders.

/exchangeInfo format:

{
  "filterType": "MAX_NUM_ALGO_ORDERS",
  "maxNumAlgoOrders": 5
}

MAX_NUM_ICEBERG_ORDERS
The MAX_NUM_ICEBERG_ORDERS filter defines the maximum number of ICEBERG orders an account is allowed to have open on a symbol. An ICEBERG order is any order where the icebergQty is > 0.

/exchangeInfo format:

{
  "filterType": "MAX_NUM_ICEBERG_ORDERS",
  "maxNumIcebergOrders": 5
}

MAX_POSITION
The MAX_POSITION filter defines the allowed maximum position an account can have on the base asset of a symbol. An account's position defined as the sum of the account's:

free balance of the base asset
locked balance of the base asset
sum of the qty of all open BUY orders
BUY orders will be rejected if the account's position is greater than the maximum position allowed.

If an order's quantity can cause the position to overflow, this will also fail the MAX_POSITION filter.

/exchangeInfo format:

{
  "filterType":"MAX_POSITION",
  "maxPosition":"10.00000000"
}

TRAILING_DELTA
The TRAILING_DELTA filter defines the minimum and maximum value for the parameter trailingDelta.

In order for a trailing stop order to pass this filter, the following must be true:

For STOP_LOSS BUY, STOP_LOSS_LIMIT_BUY,TAKE_PROFIT SELL and TAKE_PROFIT_LIMIT SELL orders:

trailingDelta >= minTrailingAboveDelta
trailingDelta <= maxTrailingAboveDelta
For STOP_LOSS SELL, STOP_LOSS_LIMIT SELL, TAKE_PROFIT BUY, and TAKE_PROFIT_LIMIT BUY orders:

trailingDelta >= minTrailingBelowDelta
trailingDelta <= maxTrailingBelowDelta
/exchangeInfo format:

    {
          "filterType": "TRAILING_DELTA",
          "minTrailingAboveDelta": 10,
          "maxTrailingAboveDelta": 2000,
          "minTrailingBelowDelta": 10,
          "maxTrailingBelowDelta": 2000
   }

Exchange Filters
EXCHANGE_MAX_NUM_ORDERS
The EXCHANGE_MAX_NUM_ORDERS filter defines the maximum number of orders an account is allowed to have open on the exchange. Note that both "algo" orders and normal orders are counted for this filter.

/exchangeInfo format:

{
  "filterType": "EXCHANGE_MAX_NUM_ORDERS",
  "maxNumOrders": 1000
}

EXCHANGE_MAX_NUM_ALGO_ORDERS
The EXCHANGE_MAX_NUM_ALGO_ORDERS filter defines the maximum number of "algo" orders an account is allowed to have open on the exchange. "Algo" orders are STOP_LOSS, STOP_LOSS_LIMIT, TAKE_PROFIT, and TAKE_PROFIT_LIMIT orders.

/exchangeInfo format:

{
  "filterType": "EXCHANGE_MAX_NUM_ALGO_ORDERS",
  "maxNumAlgoOrders": 200
}

EXCHANGE_MAX_NUM_ICEBERG_ORDERS
The EXCHANGE_MAX_NUM_ICEBERG_ORDERS filter defines the maximum number of iceberg orders an account is allowed to have open on the exchange.

/exchangeInfo format:

{
  "filterType": "EXCHANGE_MAX_NUM_ICEBERG_ORDERS",
  "maxNumIcebergOrders": 10000
}


ENUM Definitions
This will apply for both REST API and WebSocket API.

Symbol status (status)
TRADING
END_OF_DAY
HALT
BREAK

Account and Symbol Permissions (permissions)
SPOT
MARGIN
LEVERAGED
TRD_GRP_002
TRD_GRP_003
TRD_GRP_004
TRD_GRP_005
TRD_GRP_006
TRD_GRP_007
TRD_GRP_008
TRD_GRP_009
TRD_GRP_010
TRD_GRP_011
TRD_GRP_012
TRD_GRP_013
TRD_GRP_014
TRD_GRP_015
TRD_GRP_016
TRD_GRP_017
TRD_GRP_018
TRD_GRP_019
TRD_GRP_020
TRD_GRP_021
TRD_GRP_022
TRD_GRP_023
TRD_GRP_024
TRD_GRP_025
Order status (status)
Status	Description
NEW	The order has been accepted by the engine.
PENDING_NEW	The order is in a pending phase until the working order of an order list has been fully filled.
PARTIALLY_FILLED	A part of the order has been filled.
FILLED	The order has been completed.
CANCELED	The order has been canceled by the user.
PENDING_CANCEL	Currently unused
REJECTED	The order was not accepted by the engine and not processed.
EXPIRED	The order was canceled according to the order type's rules (e.g. LIMIT FOK orders with no fill, LIMIT IOC or MARKET orders that partially fill)
or by the exchange, (e.g. orders canceled during liquidation, orders canceled during maintenance)
EXPIRED_IN_MATCH	The order was expired by the exchange due to STP. (e.g. an order with EXPIRE_TAKER will match with existing orders on the book with the same account or same tradeGroupId)
Order List Status (listStatusType)
Status	Description
RESPONSE	This is used when the ListStatus is responding to a failed action. (E.g. order list placement or cancellation)
EXEC_STARTED	The order list has been placed or there is an update to the order list status.
UPDATED	The clientOrderId of an order in the order list has been changed.
ALL_DONE	The order list has finished executing and thus is no longer active.
Order List Order Status (listOrderStatus)
Status	Description
EXECUTING	Either an order list has been placed or there is an update to the status of the list.
ALL_DONE	An order list has completed execution and thus no longer active.
REJECT	The List Status is responding to a failed action either during order placement or order canceled.
ContingencyType
OCO
OTO

AllocationType
SOR

Order types (orderTypes, type)
LIMIT
MARKET
STOP_LOSS
STOP_LOSS_LIMIT
TAKE_PROFIT
TAKE_PROFIT_LIMIT
LIMIT_MAKER

Order Response Type (newOrderRespType)
ACK
RESULT
FULL
Working Floor
EXCHANGE
SOR

Order side (side)
BUY
SELL

Time in force (timeInForce)
This sets how long an order will be active before expiration.

Status	Description
GTC	Good Til Canceled
An order will be on the book unless the order is canceled.
IOC	Immediate Or Cancel
An order will try to fill the order as much as it can before the order expires.
FOK	Fill or Kill
An order will expire if the full order cannot be filled upon execution.
Rate limiters (rateLimitType)
REQUEST_WEIGHT
    {
      "rateLimitType": "REQUEST_WEIGHT",
      "interval": "MINUTE",
      "intervalNum": 1,
      "limit": 6000
    }

ORDERS
    {
      "rateLimitType": "ORDERS",
      "interval": "SECOND",
      "intervalNum": 1,
      "limit": 10
    }

RAW_REQUESTS
    {
      "rateLimitType": "RAW_REQUESTS",
      "interval": "MINUTE",
      "intervalNum": 5,
      "limit": 61000
    }

Rate limit intervals (interval)
SECOND
MINUTE
DAY

STP Modes
Read Self Trade Prevention (STP) FAQ to learn more.

NONE
EXPIRE_MAKER
EXPIRE_TAKER
EXPIRE_BOTH
DECREMENT



General API Information
The following base endpoints are available. Please use whichever works best for your setup:
https://api.binance.com
https://api-gcp.binance.com
https://api1.binance.com
https://api2.binance.com
https://api3.binance.com
https://api4.binance.com
The last 4 endpoints in the point above (api1-api4) should give better performance but have less stability.
Responses are in JSON by default. To receive responses in SBE, refer to the SBE FAQ page.
Data is returned in ascending order. Oldest first, newest last.
All time and timestamp related fields in the JSON responses are in milliseconds by default. To receive the information in microseconds, please add the header X-MBX-TIME-UNIT:MICROSECOND or X-MBX-TIME-UNIT:microsecond.
We support HMAC, RSA, and Ed25519 keys. For more information, please see API Key types.
Timestamp parameters (e.g. startTime, endTime, timestamp) can be passed in milliseconds or microseconds.
For APIs that only send public market data, please use the base endpoint https://data-api.binance.vision. Please refer to Market Data Only page.
If there are enums or terms you want clarification on, please see the SPOT Glossary for more information.
APIs have a timeout of 10 seconds when processing a request. If a response from the Matching Engine takes longer than this, the API responds with "Timeout waiting for response from backend server. Send status unknown; execution status unknown." (-1007 TIMEOUT)
This does not always mean that the request failed in the Matching Engine.
If the status of the request has not appeared in User Data Stream, please perform an API query for its status.

CHANGELOG pour l'API de Binance
Dernière mise à jour : 2025-08-07

Documentation de l'API FIX mise à jour
Limites des données du marché FIX : La limite d'abonnement a toujours été présente mais n'était pas documentée.
Sur l'ordre de traitement des messages : Reformulé et reformaté.
À partir du 08/07/2025 à 07h00 UTC , les flux WebSocket seront mis à niveau.
Lors de la mise à niveau, les connexions existantes et nouvelles peuvent être déconnectées en moins de 24 heures .
La mise à niveau peut prendre jusqu'à 2 heures ; nous nous excusons pour la gêne occasionnée.
API REST et WebSocket :

Rappel que le schéma SBE 2:0 sera retiré le 12/06/2025, 6 mois après avoir été obsolète .
Le cycle de vie SBE pour la production a été mis à jour pour refléter ce changement.
Valeur de délai d'expiration de l'API documentée et erreur sous Informations générales sur l'API pour chaque API :
RÉPARER
REPOS
WebSocket
Avis : les modifications suivantes auront lieu le 06/06/2025 à 7 h 00 UTC.

Le comportement précédent des recvWindowAPI FIX, REST et WebSocket sera augmenté par une vérification supplémentaire.
Pour récapituler, le comportement existant est :
Si le délai de réception de la requête timestampest supérieur à serverTime+ 1 seconde, celle-ci est rejetée. Ce rejet augmente les limites de messages (API FIX) et les limites IP (API REST et WebSocket), mais pas le nombre de commandes non exécutées (points de terminaison de placement de commandes de toutes les API).
Si la différence entre timestampet serverTimeà la réception de la requête est supérieure à recvWindow, la requête est rejetée. Ce rejet augmente les limites de messages (API FIX) et les limites IP (API REST et WebSocket), mais pas le nombre de commandes non exécutées (points de terminaison de placement de commandes de toutes les API).
Le contrôle supplémentaire est :
Juste avant qu'une requête ne soit transmise au moteur de correspondance, si la différence entre timestampet la valeur actuelle serverTimeest supérieure à recvWindow, la requête est rejetée. Ce rejet augmente les limites de messages (API FIX), les limites d'adresses IP (API REST et WebSocket) et le nombre de commandes non exécutées (points de terminaison de placement de commandes de toutes les API).
La documentation relative à la sécurité du timing a été mise à jour pour refléter la vérification supplémentaire.
API REST
API WebSocket
API FIX
Correction d'un bug dans le message InstrumentList de FIX Market Data <y>. Auparavant, la valeur de NoRelatedSym(146)pouvait être incorrecte.
Les fonctionnalités qui nécessitent actuellement une clé API Ed25519 seront bientôt ouvertes aux clés HMAC et RSA.
Par exemple, l'abonnement au flux de données utilisateur dans l'API WebSocket sera possible avec n'importe quel type de clé API avant que les listenKeys ne soient supprimées.
Les utilisateurs sont toujours encouragés à migrer vers les clés API Ed25519 car elles sont plus sécurisées et performantes sur Binance Spot Trading.
Plus de détails à venir.
Les poids des requêtes suivantes ont été augmentés de 1 à 4 :
API REST :PUT /api/v3/order/amend/keepPriority
API WebSocket :order.amend.keepPriority
La documentation des API REST et WebSocket a été mise à jour pour refléter ces changements.
Précision : SEQNUMdans l'API FIX, il s'agit d'un entier non signé de 32 bits avec possibilité de renouvellement. Ce SEQNUMtype de données est utilisé depuis la création de l'API FIX.
Précision sur la publication de l'Ordre de Modification, de Conservation de la Priorité et de Décrémentation du STP :

À 2025-05-07 07:00 UTC
La commande Modifier la priorité de conservation sera activée sur tous les symboles.
La décrémentation STP sera autorisée sur tous les symboles.
Le 24/04/2025 à 07h00 UTC , le champ amendAlloweddeviendra visible dans les demandes d'informations Exchange, mais la fonctionnalité ne sera pas encore activée.
SPOT Testnet a les deux fonctionnalités activées/autorisées sur tous les symboles.
Avis : Les modifications apportées à cette section seront déployées progressivement et prendront une semaine.

Nouveau code d'erreur -2039où, si vous interrogez une commande avec orderIdet origClientOrderIdet, aucune commande n'est trouvée avec cette combinaison.
Demandes concernées :
API REST :GET /api/v3/order
API WebSocket :order.status
La documentation des erreurs a également été mise à jour avec de nouveaux messages d'erreur pour le code -1034lorsque les limites de débit de connexion FIX sont dépassées. (Plus de détails sont disponibles dans la mise à jour d'hier .)
 générales
Avis : Les modifications apportées à cette section seront déployées progressivement et prendront une semaine.

Les limites de connexion aux données du marché FIX ont été augmentées de 5 à 100 le 16 janvier 2025. Cela n'était pas signalé auparavant dans le journal des modifications.
Nouveau code d'erreur -2038pour les demandes de modification de commande et de conservation de la priorité qui échouent.
Nouveaux messages pour le code d'erreur -1034.
Si le nombre de commandes non exécutées pour intervalNum:DAYest dépassé, le nombre de commandes non exécutées pour intervalNum:SECONDn'est plus incrémenté.
Auparavant, le poids de la requête pour myTrades était de 20, quels que soient les paramètres fournis. Désormais, si vous fournissez orderId, le poids de la requête est de 5.
API REST :GET /api/v3/myTrades
API WebSocket :myTrades
Modification lors de l'interrogation et de la suppression des commandes :
Lorsque ni orderIdni ne origClientOrderIdsont présents, la demande est désormais rejetée avec -1102au lieu de -1128.
Demandes concernées :
API REST :
GET /api/v3/order
DELETE /api/v3/order
API WebSocket
order.status
order.cancel
API FIX
Demande d'annulation de commande<F>
 FIX
Avis : Les changements suivants auront lieu le 21 avril 2025.

L'API FIX vérifie que la valeur EncryptMethod(98)est 0 à la connexion <A>.
Les limites de connexion de FIX Order Entry seront d'un maximum de 10 connexions simultanées par compte.
Les limites de débit de connexion sont désormais appliquées. Notez que ces limites sont vérifiées indépendamment pour le compte et l'adresse IP.
Saisie de commande FIX : 15 tentatives de connexion en 30 secondes
CORRECTION Drop Copy : 15 tentatives de connexion en 30 secondes
Données du marché FIX : 300 tentatives de connexion en 300 secondes
Les actualités <B>contiennent un compte à rebours jusqu'à la déconnexion dans le champ Titre.
Une fois cette mise à jour terminée, lorsque le serveur entrera en maintenance, un Newsmessage sera envoyé aux clients toutes les 10 secondes pendant 10 minutes . Passé ce délai, les clients seront déconnectés et leurs sessions seront fermées.
OrderCancelRequest <F>et OrderCancelRequestAndNewOrderSingle <XCN>autorisent désormais à la fois orderIdet clientOrderId.
Le schéma QuickFix pour FIX OE est mis à jour pour prendre en charge la fonctionnalité Order Amend Keep Priority et le nouveau mode STP DECREMENT.
 de données utilisateur
La réception de flux de données utilisateur sur wss://stream.binance.com:9443 à l'aide d'un listenKeyest désormais obsolète.
Cette fonctionnalité sera supprimée de nos systèmes à une date ultérieure.
Au lieu de cela, vous devez obtenir les mises à jour des données utilisateur en vous abonnant au flux de données utilisateur sur l'API WebSocket .
Cela devrait offrir des performances légèrement meilleures (latence plus faible) .
Cela nécessite l'utilisation d'une clé API Ed25519.
Dans une future mise à jour, les informations sur le point de terminaison WebSocket de base pour les flux de données utilisateur seront supprimées.
Dans une future mise à jour, les demandes suivantes seront supprimées de la documentation :
POST /api/v3/userDataStream
PUT /api/v3/userDataStream
DELETE /api/v3/userDataStream
userDataStream.start
userDataStream.ping
userDataStream.stop
La documentation User Data Stream restera une référence pour les charges utiles que vous pouvez recevoir.
 futurs
Les changements suivants auront lieu le 24 avril 2025 à 07h00 UTC :

L'option « Modifier la commande » et « Conserver la priorité » est désormais disponible. (Veuillez noter que cette fonctionnalité doit être activée pour que le symbole puisse être utilisé.)
MISE À JOUR 2025-04-21 : La date exacte à laquelle « Order Amend Keep Priority » sera activé n'a pas encore été déterminée.
Un nouveau champ amendAlloweddevient visible dans les réponses d’informations Exchange.
API REST :GET /api/v3/exchangeInfo
API WebSocket :exchangeInfo
API FIX : Nouveaux messages de saisie de commande OrderAmendKeepPriorityRequest et OrderAmendReject
API REST :PUT /api/v3/order/amend/keepPriority
API WebSocket :order.amend.keepPriority
Le mode STP DECREMENTdevient visible dans les informations d'échange si le symbole l'a configuré.
MISE À JOUR 2025-04-21 : La date exacte à laquelle DECREMENTSTP sera activé n'a pas encore été déterminée.
Au lieu de faire expirer uniquement le créateur, uniquement le preneur ou inconditionnellement les deux ordres, la décrémentation STP diminue la quantité disponible des deux ordres et augmente celle prevented quantitydes deux ordres du montant de la correspondance empêchée.
Cela fait expirer la commande dont la quantité disponible est inférieure lorsque ( filled quantity+ prevented quantity) est égal à order quantity. Les deux commandes expirent si leurs quantités disponibles sont égales. On parle de « décrémentation » car cela réduit la quantité disponible.
Comportement lors de l'interrogation et/ou de l'annulation avec orderIdet origClientOrderId/cancelOrigClientOrderId:
Le comportement lorsque les deux paramètres étaient fournis n’était pas cohérent sur tous les points de terminaison.
Par la suite, lorsque les deux paramètres sont fournis, la commande est d'abord recherchée à l'aide de son orderId, et si elle est trouvée, origClientOrderId/ cancelOrigClientOrderIdest comparée à cette commande. Si les deux conditions sont remplies, la requête aboutit. Si les deux conditions ne sont pas remplies, la requête est rejetée.
Demandes concernées :
API REST :
GET /api/v3/order
DELETE /api/v3/order
POST /api/v3/order/cancelReplace
API WebSocket :
order.status
order.cancel
order.cancelReplace
API FIX
Demande d'annulation de commande<F>
Demande d'annulation de commande et nouvelle commande unique<XCN>
Comportement lors de l'annulation avec listOrderIdet listClientOrderId:
Le comportement lorsque les deux paramètres étaient fournis n’était pas cohérent sur tous les points de terminaison.
Par la suite, lorsque les deux paramètres sont passés, la liste de commandes est d'abord recherchée à l'aide de son listOrderId, et si elle est trouvée, listClientOrderIdelle est comparée à cette liste. Si les deux conditions ne sont pas remplies, la requête est rejetée.
Demandes concernées :
API REST
DELETE /api/v3/orderList
API WebSocket
orderList.cancel
SBE : Un nouveau schéma 3:0 ( spot_3_0.xml ) est désormais disponible.
Le schéma actuel 2:1 ( spot_2_1.xml ) est désormais obsolète et sera retiré dans 6 mois conformément à notre politique d'obsolescence des schémas.
Notez que si vous essayez d'utiliser le schéma 3:0 avant sa publication, une erreur se produira.
Modifications dans le schéma 3:0 :
Prise en charge de la modification de la commande et du maintien de la priorité :
Champ ajouté amendAllowedà ExchangeInfoResponse.
Nouveaux messages OrderAmendmentsResponseetOrderAmendKeepPriorityResponse
Toutes les énumérations possèdent désormais une NON_REPRESENTABLEvariante. Celle-ci sera utilisée pour encoder de nouvelles valeurs d'énumération à l'avenir, ce qui serait incompatible avec la norme 3:0.
Nouvelle variante d'énumération DECREMENTpour selfTradePreventionModeetallowedSelfTradePreventionModes
symbolStatusvaleurs d'énumération AUCTION_MATCHet ont été supprimées PRE_TRADING.POST_TRADING
Les champs usedSor, orderCapacity, workingFloor, preventedQuantity, et matchTypene sont plus facultatifs.
Le champ orderCreationTimeest ExecutionReportEventdésormais facultatif.
Lors de l'utilisation du schéma obsolète 2:1 sur l'API WebSocket pour écouter le flux de données utilisateur :
ListStatusEventLe champ listStatusTypesera affiché tel ExecStartedqu'il aurait dû l'être Updated. Effectuez une mise à niveau vers le schéma 3:0 pour obtenir la valeur correcte.
ExecutionReportEventLe champ selfTradePreventionModesera rendu tel Nonequ'il aurait dû l'être Decrement. Cela ne se produit que lorsque executionTypeest TradePrevention.
ExecutionReportEvent le champ orderCreationTimesera rendu comme -1 lorsqu'il n'a aucune valeur.
Tous les schémas inférieurs à 3:0 ne peuvent pas représenter les réponses aux requêtes de modification de commande et de maintien de la priorité, ni toute réponse pouvant contenir le mode STP DECREMENT(par exemple, informations d'échange, passation ou annulation de commande, ou interrogation du statut de votre commande). Lorsqu'une réponse ne peut pas être représentée dans le schéma demandé, une erreur est renvoyée.
Suite à la dernière annonce de SPOT Testnet, mise à jour de l'URL dans l'API WebSocket vers la dernière URL de SPOT Testnet .
Ajout d'une clarification sur les performances de l'annulation d'une commande.
Avis : les modifications suivantes auront lieu le 13/03/2025 à 09h00 UTC
Les sessions FIX Drop Copy auront une limite de 60 messages par minute .
Les sessions FIX Market Data auront une limite de 2000 messages par minute .
La documentation de l'API FIX a été mise à jour pour refléter les changements à venir.
Les flux de données de marché SBE seront disponibles le 18 mars 2025 à 7h00 UTC. Ces flux offrent une charge utile plus faible et devraient offrir une meilleure latence que les flux JSON équivalents pour un sous-ensemble de flux de données de marché sensibles à la latence.
Flux disponibles au format SBE :
Temps réel : flux d'échanges
Temps réel : meilleur cours acheteur/vendeur
Toutes les 100 ms : diff. profondeur
Toutes les 100 ms : profondeur partielle du livre
Pour plus d'informations, veuillez vous référer aux flux de données du marché SBE .


Codes d'erreur pour Binance
Dernière mise à jour : 2025-06-11

Les erreurs se composent de deux parties : un code d'erreur et un message. Les codes sont universels, mais les messages peuvent varier. Voici la charge utile JSON de l'erreur :

{
  "code":-1121,
  "msg":"Invalid symbol."
}

10xx - 
-1000 
Une erreur inconnue s'est produite lors du traitement de la demande.
-1001 
Erreur interne ; impossible de traiter votre demande. Veuillez réessayer.
-1002 
Vous n'êtes pas autorisé à exécuter cette demande.
-1003 
Trop de demandes en file d'attente.
Poids de requête trop élevé ; la limite actuelle est de %s poids de requête par %s. Veuillez utiliser les flux WebSocket pour les mises à jour en direct afin d'éviter d'interroger l'API.
Poids de requête trop élevé ; IP bannie jusqu'à %s. Veuillez utiliser les flux WebSocket pour les mises à jour en direct afin d'éviter les bannissements.
-1006 
Une réponse inattendue a été reçue du bus de messages. Statut d'exécution inconnu.
-1007 
Délai d'attente de réponse du serveur principal dépassé. Statut d'envoi inconnu ; statut d'exécution inconnu.
-1008 
Le serveur est actuellement surchargé par d'autres requêtes. Veuillez réessayer dans quelques minutes.
-1013 
La demande est rejetée par l'API. (c'est-à-dire que la demande n'a pas atteint le moteur de correspondance.)
Les messages d'erreur potentiels peuvent être trouvés dans les échecs de filtre ou les échecs lors de la passation de commande .
-1014 
Combinaison de commandes non prise en charge.
-1015 
Trop de nouvelles commandes.
Trop de nouvelles commandes ; la limite actuelle est de %s commandes par %s.
-1016 
Ce service n'est plus disponible.
-1020 
Cette opération n'est pas prise en charge.
-1021 
L'horodatage de cette demande est en dehors de la fenêtre de réception.
L'horodatage de cette requête était en avance de 1 000 ms sur l'heure du serveur.
-1022 
La signature de cette demande n'est pas valide.
-1033 
SenderCompId(49)est actuellement utilisé. L'utilisation simultanée du même SenderCompId au sein d'un même compte n'est pas autorisée.
-1034 
Trop de connexions simultanées ; la limite actuelle est « %s ».
Trop de tentatives de connexion pour le compte ; la limite actuelle est de %s par '%s'.
Trop de tentatives de connexion depuis l'IP ; la limite actuelle est de %s par '%s'.
-1035 
Veuillez envoyer un message de déconnexion<5> pour fermer la session.
11xx - 
-1100 
Caractères illégaux trouvés dans un paramètre.
Caractères illégaux trouvés dans le paramètre '%s' ; la plage légale est '%s'.
-1101 
Trop de paramètres envoyés pour ce point de terminaison.
Trop de paramètres ; attendu '%s' et reçu '%s'.
Des valeurs en double pour un paramètre ont été détectées.
-1102 
Un paramètre obligatoire n'a pas été envoyé, était vide/nul ou mal formé.
Le paramètre obligatoire '%s' n'a pas été envoyé, était vide/nul ou mal formé.
Le paramètre '%s' ou '%s' doit être envoyé, mais les deux étaient vides/null !
Balise requise '%s' manquante.
La valeur du champ était vide ou mal formée.
« %s » contient une valeur inattendue. Ne peut être supérieur à %s.
-1103 
Un paramètre inconnu a été envoyé.
Balise non définie.
-1104 
Tous les paramètres envoyés n'ont pas été lus.
Tous les paramètres envoyés n'ont pas été lus ; '%s' paramètre(s) a été lu mais '%s' a été envoyé.
-1105 
Un paramètre était vide.
Le paramètre '%s' était vide.
-1106 
Un paramètre a été envoyé alors qu'il n'était pas nécessaire.
Paramètre '%s' envoyé lorsqu'il n'est pas requis.
Une balise '%s' a été envoyée alors qu'elle n'était pas nécessaire.
-1108 
Le paramètre '%s' a débordé.
-1111 
Le paramètre '%s' a trop de précision.
-1112 
Aucune commande sur le livre pour le symbole.
-1114 
Paramètre TimeInForce envoyé lorsqu'il n'est pas requis.
-1115 
TimeInForce non valide.
-1116 
Type de commande non valide.
-1117 
Côté invalide.
-1118 
L'ID de commande du nouveau client était vide.
-1119 
L'ID de commande client d'origine était vide.
-1120 
Intervalle non valide.
-1121 
Symbole invalide.
-1122 
Statut de symbole non valide.
-1125 
Cette listenKey n'existe pas.
-1127 
L'intervalle de recherche est trop grand.
Plus de %s heures entre startTime et endTime.
-1128 
Combinaison de paramètres optionnels non valide.
Combinaison de champs facultatifs invalide. Recommandation : « %s » et « %s » doivent être envoyés.
Les champs [%s] doivent être envoyés ensemble ou omis entièrement.
Combinaison invalide MDEntryType (269). BID et OFFER doivent être demandés ensemble.
-1130 
Données non valides envoyées pour un paramètre.
Les données envoyées pour le paramètre '%s' ne sont pas valides.
-1134 
strategyTypeétait inférieur à 1 000 000.
TargetStrategy (847)était inférieur à 1 000 000.
-1135 
Requête JSON non valide
Le JSON envoyé pour le paramètre '%s' n'est pas valide
-1139 
Type de ticker non valide.
-1145 
cancelRestrictionsdoit être soit l'un ONLY_NEWsoit l'autre ONLY_PARTIALLY_FILLED.
-1151 
Le symbole est présent plusieurs fois dans la liste.
-1152 
En-tête non valideX-MBX-SBE ; attendu <SCHEMA_ID>:<VERSION>.
-1153 
ID de schéma SBE non pris en charge ou version spécifiée dans l' X-MBX-SBEen-tête.
-1155 
SBE n'est pas activé.
-1158 
Type de commande non pris en charge dans OCO.
Si le type de commande fourni dans le aboveTypeet/ou belowTypen'est pas pris en charge.
-1160 
Le paramètre '%s' n'est pas pris en charge si aboveTimeInForce/ belowTimeInForcen'est pas GTC.
Si le type d'ordre pour la jambe supérieure ou inférieure est STOP_LOSS_LIMIT, et icebergQtyest fourni pour cette jambe, le timeInForcedoit être GTCsinon cela générera une erreur.
TimeInForce (59)doit être utilisé GTC (1)lorsqu'il MaxFloor (111)est utilisé.
-1161 
Impossible d'encoder la réponse dans le schéma SBE « x ». Veuillez utiliser le schéma « y » ou supérieur.
-1165 
Un ordre limité dans un OCO d'achat doit être inférieur.
-1166 
Un ordre limité dans un OCO de vente doit être supérieur.
-1168 
Au moins une commande OCO doit être contingente.
-1169 
Numéro d'étiquette non valide.
-1170 
La balise « %s » n'est pas définie pour ce type de message.
-1171 
La balise « %s » apparaît plus d’une fois.
-1172 
La balise '%s' a été spécifiée dans le désordre.
-1173 
Les champs du groupe « %s » se répètent dans le désordre.
-1174 
Le composant « %s » est mal renseigné dans la commande « %s ». Recommandation : « %s »
-1175 
La poursuite des numéros de séquence vers une nouvelle session n'est actuellement pas prise en charge. Les numéros de séquence doivent être réinitialisés à chaque nouvelle session.
-1176 
La connexion<A> ne doit être envoyée qu'une seule fois.
-1177 
CheckSum(10)contient une valeur incorrecte.
BeginString (8)n'est pas la première balise d'un message.
MsgType (35)n'est pas la troisième balise d'un message.
BodyLength (9)ne contient pas le nombre d'octets correct.
Seuls les caractères ASCII imprimables et SOH (Start of Header) sont autorisés.
-1178 
SenderCompId(49)Contient une valeur incorrecte. La valeur SenderCompID ne doit pas changer pendant toute la durée de la session.
-1179 
MsgSeqNum(34)contient une valeur inattendue. Attendu : « %d ».
-1180 
La connexion<A> doit être le premier message de la session.
-1181 
Trop de messages ; la limite actuelle est de « %d » messages par « %s ».
-1182 
Champs en conflit : [%s]
-1183 
L'opération demandée n'est pas autorisée dans les sessions DropCopy.
-1184 
Les sessions DropCopy ne sont pas prises en charge sur ce serveur. Veuillez vous reconnecter à un serveur DropCopy.
-1185 
Seules les sessions DropCopy sont prises en charge sur ce serveur. Reconnectez-vous au serveur de saisie de commandes ou envoyez DropCopyFlag (9406)le champ.
-1186 
L'opération demandée n'est pas autorisée dans les sessions de saisie de commandes.
-1187 
L'opération demandée n'est pas autorisée dans les sessions de données de marché.
-1188 
Nombre NumInGroup incorrect pour le groupe répétitif '%s'.
-1189 
Le groupe '%s' contient des entrées en double.
-1190 
MDReqID (262)contient un identifiant de demande d'abonnement qui est déjà utilisé sur cette connexion.
MDReqID (262)contient un identifiant de demande de désabonnement qui ne correspond à aucun abonnement actif.
-1191 
Trop d'abonnements. La connexion peut créer jusqu'à « %s » abonnements à la fois.
Un abonnement similaire est déjà actif sur cette connexion. Symbole = '%s', identifiant d'abonnement actif : '%s'.
-1194 
Valeur non valide pour l'unité de temps ; attendu soit MICROSECONDE soit MILLISECONDE.
-1196 
Un ordre stop loss dans un OCO d'achat doit être supérieur.
-1197 
Un ordre stop loss dans un OCO de vente doit être inférieur.
-1198 
Un ordre de prise de profit dans un OCO d'achat doit être inférieur.
-1199 
Un ordre de prise de profit dans un OCO de vente doit être supérieur.
-2010 
NOUVELLE_COMMANDE_REJETÉE
-2011 
ANNULER_REJETÉ
-2013 
L'ordre n'existe pas.
-2014 
Format de clé API non valide.
-2015 
Clé API, IP ou autorisations non valides pour l'action.
-2016 
Aucune fenêtre de trading n'a été trouvée pour ce symbole. Essayez plutôt ticker/24h.
-2026 
La commande a été annulée ou a expiré sans quantité exécutée il y a plus de 90 jours et a été archivée.
-2035 
L'abonnement au flux de données utilisateur est déjà actif.
-2036 
L'abonnement au flux de données utilisateur n'est pas actif.
-2039 
L'ID de commande client n'est pas correct pour cet ID de commande.

Messages pour -1010 ERROR_MSG_RECEIVED, -2010 NEW_ORDER_REJECTED, -2011 CANCEL_REJECTED et -2038 
Ce code est envoyé lorsqu'une erreur est renvoyée par le moteur de correspondance. Les messages suivants indiquent l'erreur spécifique :

Message d'erreur	Description
« Ordre inconnu envoyé. »	La commande (par orderId, clOrdId, origClOrdId) n'a pas pu être trouvée.
« Commande en double envoyée. »	Le clOrdIdest déjà utilisé.
« Le marché est fermé. »	Le symbole n'est pas négociable.
« Le solde du compte est insuffisant pour l'action demandée. »	Pas assez de fonds pour terminer l'action.
« Les ordres au marché ne sont pas pris en charge pour ce symbole. »	MARKETn'est pas activé sur le symbole.
« Les commandes Iceberg ne sont pas prises en charge pour ce symbole. »	icebergQtyn'est pas activé sur le symbole.
« Les ordres stop loss ne sont pas pris en charge pour ce symbole. »	STOP_LOSSn'est pas activé sur le symbole.
« Les ordres stop loss à cours limité ne sont pas pris en charge pour ce symbole. »	STOP_LOSS_LIMITn'est pas activé sur le symbole.
« Les ordres de prise de bénéfices ne sont pas pris en charge pour ce symbole. »	TAKE_PROFITn'est pas activé sur le symbole.
« Les ordres à cours limité de prise de bénéfices ne sont pas pris en charge pour ce symbole. »	TAKE_PROFIT_LIMITn'est pas activé sur le symbole.
« La modification de commande n'est pas prise en charge pour ce symbole. »	La modification de la commande et le maintien de la priorité ne sont pas activés sur le symbole.
« Prix * Qté est nul ou inférieur. »	price* quantityest trop bas.
« IcebergQty dépasse QTY. »	icebergQtydoit être inférieur à la quantité commandée.
« Cette action est désactivée sur ce compte. »	Contactez le support client ; certaines actions ont été désactivées sur le compte.
« Ce compte ne peut pas passer ou annuler de commandes. »	Contactez le support client ; la capacité de trading du compte est désactivée.
« Combinaison de commandes non prise en charge »	La combinaison orderType, timeInForce, stopPrice, et/ou icebergQtyn'est pas autorisée.
« L’ordre serait déclenché immédiatement. »	Le prix stop de l'ordre n'est pas valide par rapport au dernier prix négocié.
« L'annulation de la commande n'est pas valide. Vérifiez origClOrdId et orderId. »	Non origClOrdIdou orderIda été envoyé.
« La commande serait immédiatement assortie et prise. »	LIMIT_MAKERle type d'ordre correspondrait et négocierait immédiatement, et ne serait pas un ordre de fabricant pur.
« Le rapport entre les prix des commandes n'est pas correct. »	Les prix fixés dans le cadre de la OCOprésente politique enfreignent les restrictions de prix.
Pour référence :
BUY< LIMIT_MAKER priceDernier cours négocié < stopPrice
SELL: LIMIT_MAKER price> Dernier cours négocié >stopPrice
« Les ordres OCO ne sont pas pris en charge pour ce symbole »	OCOn'est pas activé sur le symbole.
« Les ordres de cotation de quantité et les ordres de marché ne sont pas pris en charge pour ce symbole. »	MARKETles commandes utilisant le paramètre quoteOrderQtyne sont pas activées sur le symbole.
« Les ordres stop suiveurs ne sont pas pris en charge pour ce symbole. »	Les commandes utilisant trailingDeltane sont pas activées sur le symbole.
« L'ordre d'annulation-remplacement n'est pas pris en charge pour ce symbole. »	POST /api/v3/order/cancelReplace(API REST) ou order.cancelReplace(API WebSocket) n'est pas activé sur le symbole.
« Ce symbole n'est pas autorisé pour ce compte. »	Le compte et le symbole n'ont pas les mêmes autorisations. (par exemple SPOT, MARGIN, etc.)
« Ce symbole est restreint pour ce compte. »	Le compte ne peut pas négocier sur ce symbole. (Par exemple, un ISOLATED_MARGINcompte ne peut pas passer SPOTd'ordres.)
« La commande n'a pas été annulée en raison de restrictions d'annulation. »	Soit cancelRestrictionsétait défini sur ONLY_NEWmais le statut de la commande ne l'était pas NEW
, soit
cancelRestrictionsétait défini sur ONLY_PARTIALLY_FILLEDmais le statut de la commande ne l'était pas PARTIALLY_FILLED.
« Le trading via l'API REST n'est pas activé. » / « Le trading via l'API WebSocket n'est pas activé. »	Une commande est en cours de passation ou un serveur n'est pas configuré pour autoriser l'accès aux TRADEpoints de terminaison.
"Le trading FIX API n'est pas activé.	La commande est passée sur un serveur FIX qui n'est pas compatible TRADE.
« La liquidité du carnet d'ordres est inférieure à LOT_SIZEla quantité minimale du filtre. »	Les ordres de marché de quantité cotée ne peuvent pas être passés lorsque la liquidité du carnet d'ordres est inférieure à la quantité minimale configurée pour le LOT_SIZEfiltre.
« La liquidité du carnet d'ordres est inférieure à MARKET_LOT_SIZEla quantité minimale du filtre. »	Les ordres de marché sur quantité cotée ne peuvent pas être passés lorsque la liquidité du carnet d'ordres est inférieure à la quantité minimale pour MARKET_LOT_SIZEle filtre.
« La liquidité du carnet d'ordres est inférieure à la quantité minimale du symbole. »	Les ordres de marché sur quantité cotée ne peuvent pas être passés lorsqu'il n'y a pas d'ordres dans le carnet.
« La modification de commande (augmentation de quantité) n'est pas prise en charge. »	newQtydoit être inférieur à la quantité commandée.
« L’action demandée ne changerait aucun état ; rejet ».	La demande envoyée n'aurait pas changé le statu quo.

(par exemple, newQtyne peut pas égaler la quantité commandée.)
Erreurs lors de la passation de commandes via 
-2021 Annulation-remplacement de commande partiellement 
Ce code est envoyé lorsque l'annulation de la commande a échoué ou que la passation de la nouvelle commande a échoué, mais pas les deux.
-2022 L'annulation-remplacement de la commande a échoué 
Ce code est envoyé lorsque l'annulation de la commande et le placement de la nouvelle commande ont échoué.

 filtre
Message d'erreur	Description
« Échec du filtre : PRICE_FILTER »	priceest trop élevé, trop bas et/ou ne respecte pas la règle de taille de tick pour le symbole.
« Échec du filtre : PERCENT_PRICE »	priceest X% trop élevé ou X% trop bas par rapport au prix moyen pondéré au cours des Y dernières minutes.
« Échec du filtre : LOT_SIZE »	quantityest trop élevé, trop bas et/ou ne respecte pas la règle de taille de pas pour le symbole.
« Échec du filtre : MIN_NOTIONAL »	price* quantityest trop bas pour être un ordre valide pour le symbole.
« Défaillance du filtre : NOTIONNEL »	price* quantityn'est pas à portée de minNotionaletmaxNotional
« Échec du filtre : ICEBERG_PARTS »	ICEBERGla commande serait divisée en trop de parties ; icebergQty est trop petit.
« Échec du filtre : MARKET_LOT_SIZE »	MARKETl'ordre quantityest trop élevé, trop bas et/ou ne respecte pas la règle de taille de pas pour le symbole.
« Échec du filtre : MAX_POSITION »	La position du compte a atteint la limite maximale définie.
Celle-ci est composée de la somme du solde de l'actif de base et de la somme de tous BUYles ordres ouverts.
« Échec du filtre : MAX_NUM_ORDERS »	Le compte a trop d'ordres ouverts sur le symbole.
"Échec du filtre : MAX_NUM_ALGO_ORDERS"	Le compte comporte trop d'ordres stop loss et/ou take profit ouverts sur le symbole.
« Échec du filtre : MAX_NUM_ICEBERG_ORDERS »	Le compte a trop d'ordres iceberg ouverts sur le symbole.
« Échec du filtre : TRAILING_DELTA »	trailingDeltan'est pas dans la plage définie du filtre pour ce type de commande.
« Échec du filtre : EXCHANGE_MAX_NUM_ORDERS »	Le compte a trop de commandes ouvertes sur la bourse.
« Échec du filtre : EXCHANGE_MAX_NUM_ALGO_ORDERS »	Le compte comporte trop d'ordres stop loss et/ou take profit ouverts sur la bourse.
« Échec du filtre : EXCHANGE_MAX_NUM_ICEBERG_ORDERS »	Le compte a trop d'ordres iceberg ouverts sur la bourse.

Réseau de test
FAQ
Comment puis-je utiliser le réseau de tests ponctuels 
Étape 1 : connectez-vous sur ce site Web et générez une clé API.

Étape 2 : Suivez la documentation officielle de l’API Spot , en remplaçant les URL des points de terminaison par les valeurs suivantes :

URL de l'API Spot	URL du réseau de test ponctuel
https://api.binance.com/api
https://api-gcp.binance.com/api
https://api1.binance.com/api
https://api2.binance.com/api
https://api3.binance.com/api
https://api4.binance.com/api
https://testnet.binance.vision/api
https://api1.testnet.binance.vision/api
wss://ws-api.binance.com/ws-api/v3
wss://ws-api.binance.com:9443/ws-api/v3
wss://ws-api.testnet.binance.vision/ws-api/v3
wss://ws-api.testnet.binance.vision:9443/ws-api/v3
wss://stream.binance.com/ws
wss://stream.binance.com:9443/ws
wss://stream.testnet.binance.vision/stream
wss://stream.testnet.binance.vision:9443/stream
wss://stream.binance.com/stream
wss://stream.binance.com:9443/stream
wss://stream.testnet.binance.vision/stream
wss://stream.testnet.binance.vision:9443/stream
wss://stream-sbe.binance.com/ws
wss://stream-sbe.binance.com:9443/ws
wss://stream-sbe.testnet.binance.vision/ws
wss://stream-sbe.testnet.binance.vision:9443/ws
wss://stream-sbe.binance.com/stream
wss://stream-sbe.binance.com:9443/stream
wss://stream-sbe.testnet.binance.vision/stream
wss://stream-sbe.testnet.binance.vision:9443/stream
tcp+tls://fix-oe.binance.com:9000
tcp+tls://fix-oe.testnet.binance.vision:9000
tcp+tls://fix-dc.binance.com:9000
tcp+tls://fix-dc.testnet.binance.vision:9000
tcp+tls://fix-md.binance.com:9000
tcp+tls://fix-md.testnet.binance.vision:9000
Puis-je utiliser les /sapipoints de terminaison sur le réseau de tests ponctuels 
Non, seuls les /apipoints de terminaison sont disponibles sur le réseau de tests ponctuels :

Points de terminaison des données de marché (API REST)
Demandes de données de marché (API WebSocket)
Flux du marché Websocket
Points de terminaison de trading (API REST)
Points de terminaison de compte (API REST)
Demandes de trading (API WebSocket)
Demandes de compte (API WebSocket)
Flux de données utilisateur
Comment faire entrer/sortir des fonds du réseau de tests ponctuels 
Tous les utilisateurs inscrits sur le réseau Spot Test reçoivent automatiquement un solde composé de différents actifs. Veuillez noter qu'il ne s'agit pas d'actifs réels et qu'ils ne peuvent être utilisés que sur le réseau Spot Test.

Tous les fonds du réseau Spot Test sont virtuels et ne peuvent pas être transférés vers/hors du réseau Spot Test.

Quelles sont les restrictions sur le réseau de tests ponctuels 
Les limites IP , les limites de taux de commande , les filtres d'échange et les filtres de symboles sur le réseau de test Spot sont généralement les mêmes que sur l'API Spot.

Tous les utilisateurs sont encouragés à interroger régulièrement l'API pour obtenir les limites de débit et les filtres les plus récents, par exemple en procédant comme suit :

curl "https://testnet.binance.vision/api/v3/exchangeInfo"
Toutes mes données ont disparu ! Que s'est-il passé 
Le réseau de tests ponctuels est périodiquement réinitialisé à un état vierge, incluant tous les ordres en attente et exécutés. Durant cette procédure de réinitialisation, tous les utilisateurs reçoivent automatiquement une nouvelle allocation de tous leurs actifs.

Ces réinitialisations se produisent environ une fois par mois et nous n'offrons pas de notification préalable à leur sujet.

Depuis août 2020, les clés API sont conservées lors des réinitialisations. Les utilisateurs n'ont plus besoin de réenregistrer de nouvelles clés API après une réinitialisation.

Quelle est la différence entre klineset uiKlines 
Sur le réseau Spot Test, ces 2 requêtes renvoient toujours les mêmes données.

Que sont les clés API RSA 
Les clés API RSA sont une alternative aux clés API HMAC-SHA-256 classiques utilisées pour authentifier vos requêtes sur l'API Spot.

Contrairement aux clés API HMAC-SHA-256 où nous générons la clé de signature secrète pour vous, avec les clés API RSA, *vous* générez une paire de clés RSA publiques+privées, nous envoyez la clé publique et signez vos demandes avec votre clé privée.

Quels types de clés RSA sont pris en charge 
Nous prenons en charge les clés RSA de toute longueur, de 2 048 bits à 4 096 bits. Nous recommandons les clés de 2 048 bits pour un bon équilibre entre sécurité et rapidité de signature.

Lors de la génération de la signature RSA, utilisez le schéma de signature PKCS#1 v1.5 . Il s'agit de la valeur par défaut avec OpenSSL. Nous ne prenons actuellement pas en charge le schéma de signature PSS.

Comment puis-je utiliser les clés API RSA 
Étape 1 : Générez la clé privée test-prv-key.pem. Ne partagez ce fichier avec personne !

openssl genrsa -out test-prv-key.pem 2048
Étape 2 : Générez la clé publique test-pub-key.pemà partir de la clé privée.

openssl rsa -in test-prv-key.pem -pubout -outform PEM -out test-pub-key.pem
La clé publique devrait ressembler à ceci :

-----BEGIN PUBLIC KEY-----
bL4DUXwR3ijFSXzcecQtVFU1zVWcSQd0Meztl3DLX42l/8EALJx3LSz9YKS0PMQW
MIICIjANBgkqhkiG9w0BAQEFAAOCAg8AMIICCgKCAgEAv9ij99RAJM4JLl8Rg47b
dJXMrv84WL1OK/gid4hCnxo083LYLXUpIqMmL+O6fmXAvsvkyMyT520Cw0ZNCrUk
WoCjGE4JZZGF4wOkWdF37JFWbDnE/GF5mAykKj+OMaECBlZ207KleQqgVzHjKuCb
hPMuBVVD3IhjBfIc7EEM438LbtayMDx4dviPWwm127jwn8qd9H3kv5JBoDfsdYMB
3k39r724CljqlAfX33GpbV2LvEkL6Da3OFk+grfN98X2pCBRz5+1N95I2cRD7o+j
wtCr+65E+Gqjo4OI60F9Gq5GDcrnudnUw13a4zwlU6W+Cy8gJ4R0CcKTc4+VhYVX
5wW2tzLVnDqvjIN8hjhgtmUv8hr19Wn+42ev+5sNtO5QAS6sJMJG5D+cpxCNhei1
Xm+1zXliaA1fvVYRqon2MdHcedFeAjzVtX38+Xweytowydcq2V/9pUUNZIzUqX7t
Zr3F+Ao3QOb/CuWbUBpUcbXfGv7AI1ozP8LRByyu6O8Z1dZNdkdjWVt83maUrIJH
jjc7jlZY9JbH6EyYV5TenjJaupvdlx72vA7Fcgevx87seog2JALAJqZQNT+t9/tm
rTUSEp3t4aINKUC1QC0CYKECAwEAAQ==
-----END PUBLIC KEY-----

Étape 3 : enregistrez votre clé publique sur le réseau de test Spot.

Lors de l'inscription, nous générerons pour vous une clé API que vous devrez mettre dans l' X-MBX-APIKEY en-tête de vos requêtes, exactement de la même manière que vous le feriez pour les clés API HMAC-SHA-256.

Étape 4 : lorsque vous envoyez une demande au réseau de test ponctuel, signez la charge utile à l’aide de votre clé privée.

Voici un exemple de script Bash permettant de publier une nouvelle commande et de signer la requête avec OpenSSL. Vous pouvez l'adapter à votre langage de programmation préféré :

#!/usr/bin/env bash

# Set up authentication:
API_KEY="put your own API Key here"
PRIVATE_KEY_PATH="test-prv-key.pem"

# Set up the request:
API_METHOD="POST"
API_CALL="api/v3/order"
API_PARAMS="symbol=BTCUSDT&side=SELL&type=LIMIT&timeInForce=GTC&quantity=1&price=0.2"

# Sign the request:
timestamp=$(date +%s000)
api_params_with_timestamp="$API_PARAMS&timestamp=$timestamp"
signature=$(echo -n "$api_params_with_timestamp" \
            | openssl dgst -sha256 -sign "$PRIVATE_KEY_PATH" \
            | openssl enc -base64 -A)

# Send the request:
curl -H "X-MBX-APIKEY: $API_KEY" -X "$API_METHOD" \
    "https://testnet.binance.vision/$API_CALL?$api_params_with_timestamp" \
    --data-urlencode "signature=$signature"


Que sont les clés API Ed25519 
Les clés API Ed25519 sont une alternative aux clés API RSA , utilisant la cryptographie asymétrique pour authentifier vos requêtes sur l'API Spot.

Comme les clés API RSA, les clés Ed25519 sont asymétriques : vous générez une paire de clés, partagez la clé publique avec Binance et utilisez votre clé privée pour signer les requêtes.

Pourquoi utiliser Ed25519 au lieu des clés API RSA 
Le schéma de signature numérique Ed25519 offre une sécurité comparable aux clés RSA 3072 bits, tout en ayant des signatures beaucoup plus petites et plus rapides à calculer :

Type de clé API	Taille de la signature	Opération de signature
HMAC-SHA-256	64 octets	0,00 ms
Ed25519	88 octets	0,03 ms
RSA (2048 bits)	344 octets	0,55 ms
RSA (4096 bits)	684 octets	3,42 ms
Comment puis-je utiliser les clés API Ed25519 
Étape 1 : Générez la clé privée test-prv-key.pem. Ne partagez ce fichier avec personne !

openssl genpkey -algorithm ed25519 -out test-prv-key.pem
Étape 2 : Calculez la clé publique test-pub-key.pemà partir de la clé privée.

clé de protection openssl -pubout -in test-prv-key.pem -out test-pub-key.pem
La clé publique devrait ressembler à ceci :

-----BEGIN PUBLIC KEY-----
MCowBQYDK2VwAyEACeCSz7VJkh3Bb+NF794hLMU8fLB9Zr+/tGMdVKCC2eo=
-----END PUBLIC KEY-----

Étape 3 : enregistrez votre clé publique sur le réseau de test Spot.

Lors de votre inscription, nous générerons une clé API. Veuillez l'insérer dans l' X-MBX-APIKEYen-tête de vos requêtes, exactement comme pour les autres types de clés API.

Étape 4 : lorsque vous envoyez une demande au réseau de test ponctuel, signez la charge utile à l’aide de votre clé privée.

Voici un exemple en Python qui publie une nouvelle commande signée avec la clé Ed25519. Vous pouvez l'adapter à votre langage de programmation préféré.

#!/usr/bin/env python3

import base64
import requests
import time
from cryptography.hazmat.primitives.serialization import load_pem_private_key

# Set up authentication
API_KEY='put your own API Key here'
PRIVATE_KEY_PATH='test-prv-key.pem'

# Load the private key.
# In this example the key is expected to be stored without encryption,
# but we recommend using a strong password for improved security.
with open(PRIVATE_KEY_PATH, 'rb') as f:
    private_key = load_pem_private_key(data=f.read(),
                                       password=None)

# Set up the request parameters
params = {
    'symbol':       'BTCUSDT',
    'side':         'SELL',
    'type':         'LIMIT',
    'timeInForce':  'GTC',
    'quantity':     '1.0000000',
    'price':        '0.20',
}

# Timestamp the request
timestamp = int(time.time() * 1000) # UNIX timestamp in milliseconds
params['timestamp'] = timestamp

# Sign the request
payload = '&'.join([f'{param}={value}' for param, value in params.items()])
signature = base64.b64encode(private_key.sign(payload.encode('ASCII')))
params['signature'] = signature

# Send the request
headers = {
    'X-MBX-APIKEY': API_KEY,
}
response = requests.post(
    'https://testnet.binance.vision/api/v3/order',
    headers=headers,
    data=params,
)
print(response.json())

Glossaire de l'API SPOT
Avertissement : Ce glossaire concerne uniquement l'implémentation de l'API SPOT. La définition de ces termes peut différer pour les contrats à terme, les options et les autres API de Binance.

ACK

newOrderRespTypeenum. Représente un type de réponse de commande en JSON où seuls les champs suivants sont émis : symbol, orderId, orderListId, clientOrderId, et transactTime.
aggTrade/Commerce global

Agrégation d'une ou plusieurs transactions individuelles provenant du même ordre de preneur qui ont été exécutées au même moment et au même prix.
allocation

Transfert d'actifs de la bourse vers votre compte (par exemple, lorsqu'un ordre est exécuté par SOR au lieu d'être négocié directement).
allocationId

Identifiant unique d'une allocation sur un symbole.
allocationType

Voir AllocationType
askPrice

Dans les réponses du téléscripteur : le prix le plus bas du SELLcôté.
askQty

Dans les réponses du téléscripteur : quantité totale offerte au prix le plus bas du SELLcôté.
asks

Commandes à SELLcôté.
avgPrice

Représente le prix moyen pondéré en fonction du volume pour un intervalle défini de minutes.
baseAsset

Le premier actif du symbole (par exemple BTC, le base assetsymbole BTCUSDT), qui représente l'actif acheté et vendu (le quantity).
baseAssetPrecision

Un champ trouvé dans Exchange Information qui indique le nombre de décimales autorisées sur le baseAsset.
baseCommissionPrecision

Un champ trouvé dans les informations d'échange qui représente le nombre de décimales sur lesquelles la commission d'actif de base sera calculée.
bidPrice

Dans les réponses du ticker : le prix le plus élevé du BUYcôté.
bidQty

Dans les réponses du ticker : quantité totale offerte au prix le plus élevé du BUYcôté.
bids

Commandes à BUYcôté.
BREAK

Le statut de trading du symbole qui le représente n'est pas disponible pour le trading, ce qui peut se produire pendant les périodes d'indisponibilité prévues. Les données de marché ne sont pas générées pendant cette période BREAK.
BUY

Une énumération dans le sideparamètre lorsqu'un utilisateur souhaite acheter un actif (par exemple BTC).
CANCELED

Commande statusindiquant que la commande a été annulée par l'utilisateur.
cancelReplaceMode

Paramètre utilisé dans les commandes Annuler et Remplacer qui définissent si le placement de la nouvelle commande doit se poursuivre si la demande d'annulation échoue.
clientOrderId

Un champ, qui peut être défini par l'utilisateur, dans la réponse JSON pour les demandes de passation de commande afin d'identifier la commande nouvellement passée.
commission

Les frais qui ont été payés sur une transaction.
commissionAsset

L'actif sur lequel les frais de commission ont été déduits.
ID de la commande au comptoir

Un champ dans les rapports d'exécution du flux de données utilisateur qui indique l'ordre de contrepartie dans une correspondance empêchée.
Symbole de compteur

Un champ dans les rapports d'exécution du flux de données utilisateur qui indique le symbole de l'ordre de contrepartie dans une correspondance empêchée.
cummulativeQuoteQty

L'accumulation du price* qtypour chaque remplissage d'une commande.
Source des données

Spécifie où le point de terminaison ou la demande récupère ses données.
executedQty

Le champ qui indique la quantité qui a été remplie dans une commande.
EXPIRED

Ordre statusindiquant que la commande a été annulée conformément aux règles du type de commande ou par la bourse.
EXPIRED_IN_MATCH

Ordre statusindiquant que la commande a été annulée par la bourse en raison du STP. (par exemple, une commande avec EXPIRE_TAKERcorrespondra aux commandes existantes sur le carnet avec le même compte ou le même tradeGroupId)
filters

Définit les règles de négociation sur la bourse.
FOK/ Remplir ou tuer

timeInForceénumération où l'ordre ne sera pas exécuté et expirera si l'ordre ne peut pas être entièrement exécuté.
free

Le montant d'un actif dans le solde d'un utilisateur qui peut être utilisé pour effectuer des transactions ou des retraits.
FULL

newOrderRespTypeenum. Représente un type de réponse de commande en JSON, où toutes les informations de commande sont émises, y compris fillsle champ de commandes.
GTC/ Bon jusqu'à annulation

timeInForceénumération où la commande restera active jusqu'à ce qu'elle soit annulée ou entièrement exécutée.
HALT

Le statut de trading du symbole qui le représente n'est pas disponible pour le trading, ce qui peut se produire en cas d'interruption d'activité. Les données de marché sont toujours générées pendant cette période HALT.
intervalNum

Décrit la durée de l'intervalle (par exemple, si intervalest SECONDet intervalNumest 5, alors cela sera interprété comme toutes les 5 secondes).
IOC/ Immédiat ou Annulé

timeInForceénumération où la commande essaie de remplir autant que possible, et la quantité restante non remplie expirera.
isBestMatch

Champ dans la réponse JSON qui détermine si le prix de la transaction était le meilleur disponible sur la bourse.
isBuyerMaker

Champ dans la réponse JSON qui indique si le côté acheteur (l'acheteur) était également le teneur de marché (le fabricant).
isWorking

Champ dans le JSON qui indique si la commande a commencé à fonctionner sur le carnet de commandes.
kline

Identifie le cours d'ouverture, de clôture, le plus haut, le plus bas, le volume de transactions et d'autres données de marché d'un symbole à un moment précis et pendant une durée déterminée. Également appelé chandelier.
Dernière quantité empêchée

Quantité de commande expirée en raison du STP.
lastPrice

Prix de la dernière transaction.
lastQty

Quantité totale échangée au lastPrice.
LIMIT

un typeordre dont le prix d'exécution ne sera pas inférieur au prix fixé. Le prix d'exécution est limité au prix fixé ou supérieur.
LIMIT_MAKER

Un typeordre où l'ordre ne peut être qu'un ordre de fabricant (c'est-à-dire que l'ordre ne peut pas correspondre et prendre immédiatement).
limitClientOrderId

Un paramètre utilisé pour placer des commandes OCO qui identifie la LIMIT_MAKERpaire de la commande OCO.
listClientOrderId

Un paramètre utilisé pour passer des commandes OCO qui identifie la paire de commandes.
listenKey

Clé individuelle utilisée sur User Data Stream pour obtenir des mises à jour en direct sur le compte associé.
locked

Le montant d'un actif dans les soldes d'un utilisateur qui sont actuellement verrouillés dans des ordres ouverts et d'autres services par la plateforme.
MARKET

Un typeordre dans lequel l'utilisateur achète ou vend un actif aux meilleurs prix et liquidités disponibles jusqu'à ce que l'ordre soit entièrement exécuté ou que la liquidité du carnet d'ordres soit épuisée.
Moteur de correspondance

Cela peut faire référence à une source de données dans la documentation, ce qui signifie que la réponse provient du moteur.
Ou alors on parle du système qui gère toutes les demandes et fait correspondre les commandes.
Type de correspondance

Champ dans la réponse à l'ordre ou le rapport d'exécution indiquant si l'ordre a été exécuté par le SOR
Mémoire

Source de données d'où provient la réponse de la mémoire interne ou du cache de l'API.
NEW

Commande statusoù une commande a été envoyée avec succès au moteur de correspondance.
newClientOrderId

Paramètre utilisé dans l'API SPOT pour attribuer le clientOrderIdpour la commande passée ou le message d'annulation.
valeur notionnelle

La valeur price* qty.
OCO

Type d'ordre One-Cancels-the-Other composé d'une paire d'ordres (par exemple STOP_LOSSou STOP_LOSS_LIMITassocié à un LIMIT_MAKERordre) avec la condition que si l'un des ordres s'exécute, l'autre expire automatiquement.
Commander Modifier Garder la priorité

Voir l'ordre Modifier Garder la priorité
Carnet de commandes

Liste des offres et demandes ouvertes pour un symbole.
Liste de commandes

Plusieurs commandes regroupées en une seule unité. Voir OCOet/ouOTO
orderId

Un champ dans la réponse de commande qui identifie de manière unique la commande sur un symbole.
origQty

L'original quantityqui a été envoyé lors de la passation de la commande.
origClientOrderId

Champ utilisé lors de l'annulation ou de l'interrogation d'une commande en fournissant le clientOrderId.
OTO

Type d'ordre « L'un déclenche l'autre » qui comporte un ordre en cours et un ordre en attente.
Lorsque l'ordre de travail est entièrement rempli, la commande en attente est automatiquement passée.
OTOCO

L'ordre One-Triggers-a-One-Cancels-the-Other a un ordre en cours et une paire OCO pour les ordres en attente.
Lorsque l'ordre de travail est entièrement rempli, la paire OCO en attente est automatiquement placée.
PARTIALLY_FILLED

Ordre statusindiquant qu'une partie de la commande a été partiellement exécutée.
Commande en attente

Une commande dans une liste de commandes qui n'est placée dans le carnet de commandes que lorsque l'ordre de travail correspondant est entièrement rempli.
Une seule liste de commandes peut contenir soit une seule commande en attente, soit 2 commandes en attente formant un OCO.
Dans le cas d'une commande unique, presque tous les types de commandes sont pris en charge, à l'exception des MARKETcommandes utilisant quoteOrderQty.
PENDING_NEW

Ordre statusindiquant que les commandes en attente d'une liste de commandes ont été acceptées par le moteur, mais ne sont pas encore placées dans le carnet de commandes.
Prix d'exécution empêché

Un champ dans les rapports d'exécution du flux de données utilisateur affiche le prix d'une auto-transaction empêchée. Voir STP .
Quantité d'exécution empêchée

Champ du flux de données utilisateur indiquant la quantité d'auto-échanges empêchés. Voir STP .
Quantité de devis d'exécution empêchée

Champ du flux de données utilisateur indiquant la quantité de cotation de l'auto-négociation empêchée. Voir STP .
preventedQuantity

La quantité commandée a expiré en raison d'événements STP.
Match empêché

Lorsque la ou les commandes expirent en raison du STP, une « correspondance empêchée » enregistre l'événement.
preventedMatchId

Lorsqu'il est utilisé en combinaison avec symbol, peut être utilisé pour interroger une correspondance empêchée de la commande expirée.
quantity

Paramètre utilisé pour spécifier le montant base assetà acheter ou à vendre.
quoteAsset

Le deuxième actif du symbole (par exemple USDT, le quote assetsymbole BTCUSDT) qui représente l'actif utilisé pour coter les prix (le price).
quoteAssetPrecision

Un champ trouvé dans Exchange Information qui indique le nombre de décimales autorisées sur le quoteAsset.
quoteCommissionPrecision

Un champ trouvé dans les informations d'échange qui représente le nombre de décimales auxquelles la commission de cotation des actifs sera calculée.
quoteOrderQty

MARKETparamètre d'ordre qui spécifie le montant de l'actif de cotation que l'on souhaite dépenser/recevoir dans un « ordre MARCHÉ inversé ».
quoteQty

price* qty; la valeur notionnelle.
recvWindow

Paramètre dans les API qui peut être utilisé pour spécifier le nombre de millisecondes après lequel timestampla demande est valide.
RESULT

newOrderRespTypeenum. Représente un type de réponse de commande en JSON, où toutes les informations de commande sont émises, à l'exception du fillschamp de commande.
MARKETOrdre inverse

Une MARKETcommande qui est spécifiée en utilisant le quoteOrderQtyau lieu du quantity.
Prévention du commerce libre-service (STP)

La prévention des échanges automatiques est une fonctionnalité qui empêche les ordres des utilisateurs, ou ceux de ces derniers, tradeGroupIdde correspondre aux leurs. Consultez la FAQ STP pour en savoir plus.
selfTradePreventionMode

Un paramètre utilisé pour spécifier ce que le système fera si un ordre peut entraîner une auto-négociation.
SELL

Une énumération sideutilisée lorsqu'un utilisateur souhaite vendre un actif (par exemple BTC).
Routage intelligent des commandes (SOR)

Le routage intelligent des ordres utilise des actifs de cotation interchangeables pour améliorer la liquidité. Consultez la FAQ SOR pour en savoir plus.
specialCommissionForOrder/specialCommission

Voir la FAQ de la Commission
SPOT

Il s’agit de distinguer un type de trading, où l’achat et la livraison d’un actif sont effectués immédiatement.
standardCommissionForOrder/standardCommission

Voir la FAQ de la Commission
stopClientOrderId

Un paramètre utilisé pour placer des commandes OCO qui identifie la STOP_LOSSou STOP_LOSS_LIMITles paires de commandes OCO.
stopPrice

Le prix utilisé dans les ordres algorithmiques (par exemple STOP_LOSS, TAKE_PROFIT) qui détermine quand un ordre sera déclenché pour être placé dans le carnet d'ordres.
Le prix utilisé dans les ordres algorithmiques suiveurs (par exemple STOP_LOSS, TAKE_PROFIT) pour déterminer quand le suivi des prix suiveurs commence.
STOP_LOSS

Un typeordre algorithmique où une fois que le prix du marché atteint le stopPrice, un MARKETordre est placé sur le carnet d'ordres.
STOP_LOSS_LIMIT

Un typeordre algorithmique où une fois que le prix du marché atteint le stopPrice, un LIMITordre est placé sur le carnet d'ordres.
strategyId

Valeur numérique arbitraire identifiant l'ordre dans une stratégie de commande.
strategyType

Valeur numérique arbitraire identifiant la stratégie de commande.
symbol

Une paire de trading, composée de a base assetet de a quote asset. (par exemple BTCUSDT et BNBBTC)
TAKE_PROFIT

Un typeordre algorithmique où une fois que le prix du marché atteint le stopPrice, un MARKETordre est placé sur le carnet d'ordres.
TAKE_PROFIT_LIMIT

Un typeordre algorithmique où une fois que le prix du marché atteint le stopPrice, un LIMITordre est placé sur le carnet d'ordres.
taxCommissionForOrder/taxCommission

Voir la FAQ de la Commission
ticker

Signale le changement de prix et d'autres données du fabricant d'un symbole dans un certain intervalle glissant.
time

Pour les requêtes de transactions/allocations : l'heure à laquelle les transactions/allocations ont été exécutées.
Pour les requêtes de commande : l'heure à laquelle les commandes ont été créées.
timeInForce

Détermine le comportement du preneur d'un ordre, si un ordre peut être un ordre de fabricant et combien de temps l'ordre restera dans le carnet d'ordres avant d'expirer.
Les énumérations prises en charge sont GTC, IOC, et FOK.
tradeGroupId

Groupe de comptes appartenant au même « groupe commercial ».
TRADING

Statut commercial où les commandes peuvent être passées.
trailingDelta

Paramètre d'ordre Trailing Stop qui spécifie le changement de prix delta requis avant l'activation de l'ordre.
trailingTime

L'heure à laquelle l'ordre suiveur est désormais actif et suit les variations de prix.
transactTime

Heure de mise à jour de la commande : passée, exécutée ou annulée. Ce champ (ainsi que tous les champs liés à l'horodatage) sera exprimé en millisecondes par défaut dans les réponses JSON.
uiKlines

Données de chandelier modifiées optimisées pour la présentation de graphiques en chandeliers.
updateTime

Dernière mise à jour de la commande. Ce champ (ainsi que tous les champs liés à l'horodatage) sera exprimé en millisecondes par défaut dans les réponses JSON.
Flux de données utilisateur

Flux WebSocket utilisé pour obtenir des informations en temps réel sur le compte d'un utilisateur. (par exemple, modifications des soldes, mises à jour des commandes, etc.) Lisez Flux de données utilisateur pour en savoir plus.
usedSor

Indique si la commande a été passée via SOR .
weightedAveragePrice

Le prix moyen pondéré en fonction du volume au cours des x dernières minutes.
workingFloor

Un champ qui détermine si la commande est exécutée par le SOR ou par le carnet de commandes auquel la commande a été soumise.
Ordre de marche

Un ordre dans une liste d'ordres qui est immédiatement placé dans le carnet d'ordres et qui déclenchera le placement d'un ou plusieurs ordres en attente lorsqu'il sera entièrement exécuté.
Un ordre dans une liste de commandes qui se compose toujours d'un seul ordre LIMITou .LIMIT_MAKER
workingTime

L'heure à laquelle la commande a commencé à fonctionner sur le carnet de commandes.
X-MBX-ORDER-COUNT-XX

En-tête de réponse émis lorsqu'un utilisateur passe une commande, indiquant le nombre de commandes actuel pour l'intervalle XX pour ce compte.
X-MBX-USED-WEIGHT-XX

En-tête de réponse émis lorsqu'un utilisateur envoie une requête à l'API, indiquant le poids de requête actuellement utilisé pour l'intervalle XX par l'IP de l'utilisateur.

Ordres indexés
Clause de non-responsabilité :

Cette explication s'applique uniquement à l'échange SPOT.
Les symboles et valeurs utilisés ici sont fictifs et n'impliquent rien sur la configuration réelle de l'échange en direct.
Par souci de simplicité, les exemples de ce document n'incluent pas de commission.
Que sont les ordres indexés
Les ordres indexés sont essentiellement des ordres à cours limité dont le prix est dérivé du carnet d'ordres.

Par exemple, au lieu d'utiliser un prix spécifique (par exemple, VENDRE 1 BTC pour au moins 100 000 USDC), vous pouvez envoyer des ordres tels que « VENDRE 1 BTC au meilleur prix demandé » pour mettre votre ordre en file d'attente après les ordres du carnet au prix le plus élevé, ou « ACHETER 1 BTC pour 100 000 USDT ou meilleure offre, IOC » pour sélectionner les vendeurs au prix le plus bas, et uniquement à ce prix.

Les ordres indexés offrent aux teneurs de marché un moyen d'obtenir le meilleur prix avec une latence minimale, tandis que les utilisateurs de détail peuvent obtenir des exécutions rapides au meilleur prix avec un glissement minimal.

Les ordres indexés sont également connus sous le nom d'ordres « meilleure offre-offre » ou ordres BBO.

Comment puis-je envoyer une commande fixe
Veuillez vous référer au tableau suivant :

API	Demande	Paramètres
API REST	POST /api/v3/order	
pegPriceType:

PRIMARY— meilleur prix du même côté du carnet de commandes
MARKET— meilleur prix du côté opposé du carnet de commandes
pegOffsetTypeet pegOffsetValue PRICE_LEVEL— compensé par les niveaux de prix existants, plus profondément dans le carnet de commandes

Pour les listes de commandes : (Veuillez consulter la documentation de l’API pour plus de détails.)

OCO utilise les préfixes above*et below*.
OTO utilise les préfixes working*et pending*.
OTOCO utilise les préfixes working*, pendingAbove*, et pendingBelow*.
POST /api/v3/orderList/*
POST /api/v3/cancelReplace
API WebSocket	order.place
orderList.place.*
order.cancelReplace
API FIX	NouvelleCommandeSingle<D>	OrdType=PEGGED, <PegInstructions>bloc composant, PeggedPricechamp.
Nouvelle liste de commandes<E>
Demande d'annulation de commande et nouvelle commande unique<XCN>
Actuellement, Smart Order Routing (SOR) ne prend pas en charge les ordres indexés.

Cet exemple de réponse d'API REST montre que pour les commandes indexées, peggedPricereflète le prix sélectionné, tandis priceque est le prix de la commande d'origine (zéro s'il n'est pas défini).

{
  "symbol": "BTCUSDT",
  "orderId": 18,
  "orderListId": -1,
  "clientOrderId": "q1fKs4Y7wgE61WSFMYRFKo",
  "transactTime": 1750313780050,
  "price": "0.00000000",
  "pegPriceType": "PRIMARY_PEG",
  "peggedPrice": "0.04000000",
  "origQty": "1.00000000",
  "executedQty": "0.00000000",
  "origQuoteOrderQty": "0.00000000",
  "cummulativeQuoteQty": "0.00000000",
  "status": "NEW",
  "timeInForce": "GTC",
  "type": "LIMIT",
  "side": "BUY",
  "workingTime": 1750313780050,
  "fills": [],
  "selfTradePreventionMode": "NONE"
}

Quels types d'ordres prennent en charge les ordres indexés
Tous les types de commandes, à l'exception des MARKETcommandes, sont pris en charge par cette fonctionnalité.

Étant donné que les ordres STOP_LOSSet TAKE_PROFITpassent un MARKETordre une fois que la condition d'arrêt est remplie, ces types d'ordres ne peuvent pas être indexés.

 à cours limité
Les ordres à cours limité indexés entrent immédiatement sur le marché au meilleur prix actuel :

LIMIT
Avec pegPriceType=PRIMARY_PEGseulement timeInForce=GTCest autorisé.
LIMIT_MAKER
Seul pegPriceType=PRIMARY_PEGest autorisé.
 stop-limite
Les ordres stop-limit indexés entrent sur le marché au meilleur prix lorsque le mouvement de prix déclenche l'ordre stop (via le prix stop ou le stop suiveur) :

STOP_LOSS_LIMIT
TAKE_PROFIT_LIMIT
Autrement dit, les ordres stop utilisent le meilleur prix au moment de leur déclenchement, qui est différent du prix au moment du placement de l'ordre stop. Seul le prix limite peut être indexé, et non le prix stop.

Les listes de commandes OCO peuvent utiliser des instructions de rattachement.

Tout ordre dans OCO peut être indexé : à la fois sur les ordres supérieurs et inférieurs, ou sur un seul d'entre eux.
Les ordres indexés entrent au meilleur prix lorsqu'ils sont placés sur le carnet :
LIMIT_MAKERla commande entre immédiatement au meilleur prix actuel
STOP_LOSS_LIMITet TAKE_PROFIT_LIMITentrez au meilleur prix lorsqu'ils sont déclenchés
STOP_LOSSet TAKE_PROFITles commandes ne peuvent pas être indexées.
OTO et 
Les listes de commandes OTO peuvent également utiliser des instructions de rattachement.

Toute commande dans OTO peut être indexée : à la fois les commandes en cours et en attente, ou une seule d'entre elles.
L'ordre de travail indexé entre immédiatement au meilleur prix actuel.
L'ordre à cours limité en attente est entré au meilleur prix une fois l'ordre de travail exécuté.
L'ordre stop-limit en attente indexé entre au meilleur prix lorsqu'il est déclenché.
Les listes d'ordres OTOCO peuvent également contenir des ordres indexés, similaires à OTO et OCO.

Quels symboles permettent les ordres indexés
Veuillez consulter les demandes d'informations sur la bourse et rechercher le champ pegInstructionsAllowed. Si ce champ est défini sur « true », les ordres indexés peuvent être utilisés avec le symbole.

Quels filtres sont applicables aux ordres indexés
Les commandes indexées doivent passer tous les filtres applicables avec le prix sélectionné :

PRICE_FILTER
PERCENT_PRICEetPERCENT_PRICE_BY_SIDE
NOTIONALet MIN_NOTIONAL(considérant le quantity)
Si un ordre indexé spécifie price, il doit passer la validation à la fois à priceet à peggedPrice.

Les ordres contingents indexés ainsi que les ordres indexés en attente des listes d'ordres OTO sont (re)validés au moment du déclenchement et peuvent être rejetés ultérieurement.

Précédent
Glossaire des spots


Taux de commission
Clause de non-responsabilité:

Les commissions et les prix utilisés ici sont fictifs et n'impliquent rien sur la configuration réelle de l'échange en direct.
Ceci s'applique uniquement à l'échange SPOT.
Que sont les taux de commission
Il s'agit des taux qui déterminent la commission à payer sur les transactions lorsque votre ordre est exécuté pour n'importe quel montant.

Quels sont les différents types de tarifs 
Il existe 3 types :

standardCommission- Taux de commission standard sur les transactions de la commande.
taxCommission- Taux de commission fiscale sur les transactions issues de la commande.
specialCommission- Commission supplémentaire qui sera ajoutée dans des circonstances particulières.
Le taux de commission standard peut être réduit, en fonction des promotions pour des paires de trading spécifiques, des remises applicables, etc.

Comment puis-je connaître les taux de commission 
Vous pouvez les trouver en utilisant les requêtes suivantes :

API REST :GET /api/v3/account/commission

API WebSocket :account.commission

Vous pouvez également connaître les taux de commission d'une transaction à partir d'une commande en utilisant les demandes de commande de test avec computeCommissionRates.


Quelle est la différence entre la réponse à l'envoi d'une commande test et computeCommissionRatesla réponse à la demande de taux de commission
Une commande test avec computeCommissionRatesdes taux de commission détaillés pour cette commande spécifique :

{
  "standardCommissionForOrder": {
    "maker": "0.00000050",
    "taker": "0.00000060"
  },
  "specialCommissionForOrder": {
    "maker": "0.05000000",
    "taker": "0.06000000"
  },
  "taxCommissionForOrder": {
    "maker": "0.00000228",
    "taker": "0.00000230"
  },
  "discount": {
    "enabledForAccount": true,
    "enabledForSymbol": true,
    "discountAsset": "BNB",
    "discount": "0.25000000"
  }
}

Remarque : les commissions acheteur/vendeur ne sont pas affichées séparément, car elles sont déjà prises en compte en fonction du côté de la commande.

En revanche, l'interrogation des taux de commission renvoie vos taux de commission actuels pour le symbole de votre compte.

{
  "symbol": "BTCUSDT",
  "standardCommission": {
    "maker": "0.00000040",
    "taker": "0.00000050",
    "buyer": "0.00000010",
    "seller": "0.00000010"
  },
  "specialCommission": {
    "maker": "0.04000000",
    "taker": "0.05000000",
    "buyer": "0.01000000",
    "seller": "0.01000000"
  },
  "taxCommission": {
    "maker": "0.00000128",
    "taker": "0.00000130",
    "buyer": "0.00000100",
    "seller": "0.00000100"
  },
  "discount": {
    "enabledForAccount": true,
    "enabledForSymbol": true,
    "discountAsset": "BNB",
    "discount": "0.25000000"
  }
}

Comment la commission est-elle calculée 
En utilisant un exemple de configuration de commission :

{
  "symbol": "BTCUSDT",
  "standardCommission": {
    "maker": "0.00000010",
    "taker": "0.00000020",
    "buyer": "0.00000030",
    "seller": "0.00000040"
  },
  "specialCommission": {
    "maker": "0.01000000",
    "taker": "0.02000000",
    "buyer": "0.03000000",
    "seller": "0.04000000"
  },
  "taxCommission": {
    "maker": "0.00000112",
    "taker": "0.00000114",
    "buyer": "0.00000118",
    "seller": "0.00000116"
  },
  "discount": {
    "enabledForAccount": true,
    "enabledForSymbol": true,
    "discountAsset": "BNB",
    "discount": "0.25000000"
  }
}

Si vous avez passé un ordre avec les paramètres suivants qui a été immédiatement et entièrement exécuté en une seule transaction :

Paramètre	Valeur
symbole	BTCUSDT
prix	35 000
quantité	0,49975
côté	VENDRE
taper	MARCHÉ
Étant donné que vous avez vendu du BTC contre de l'USDT, la commission sera payée soit en USDT, soit en BNB.

Lors du calcul de la commission standard, le montant reçu est multiplié par la somme des taux.

Étant donné que cet ordre est latéral SELL, le montant reçu est la valeur notionnelle. (Pour les ordres latéraux BUY, le montant reçu serait quantity.) Le type d'ordre était MARKET, ce qui en fait l'ordre du preneur pour la transaction.

Standard Commission = Notional value * (taker + seller)
                    = (35000 * 0.49975) * (0.00000020 + 0.00000040)
                    = 17491.25000000 * 0.00000060
                    = 0.01049475 USDT

La commission fiscale (le cas échéant) est calculée de la même manière :

Tax commission = Notional value * (taker + seller)
               = (35000 * 0.49975) * (0.00000114 + 0.00000116)
               = 17491.25000000 * 0.00000230
               = 0.04022988 USDT

La commission spéciale (le cas échéant) est calculée comme suit :

Special commission = Notional value * (taker + seller)
               = (35000 * 0.49975) * (0.02000000 + 0.04000000)
               = 17491.25000000 * 0.06000030
               = 1049.47500000 USDT

Si vous ne payez pas en BNB, la commission totale est additionnée et déduite du montant reçu de USDT.

Étant donné que enabledforAccountet enabledForSymbolmoins discountest défini sur true, cela signifie que la commission sera payée en BNB en supposant que vous disposez d'un solde suffisant.

Si vous payez avec BNB, la commission standard sera réduite en fonction du discount.

Tout d'abord, la commission standard et la commission fiscale seront converties en BNB en fonction du taux de change. Dans cet exemple, supposons que 1 BNB = 260 USDT.

Standard commission (Discounted and in BNB) = (Standard commission * BNB exchange rate) * discount
                                            = (0.01049475 * 1/260) * 0.25
                                            = 0.000040364 * 0.25
                                            = 0.000010091


Veuillez noter que la remise ne s'applique pas aux commissions fiscales ou aux commissions spéciales .

Tax Commission (in BNB) = Tax commission * BNB exchange rate
                        = 0.04022988 * (1/260)
                        = 0.00015473

Special Commission (in BNB) = Special commission * BNB exchange rate
                        = 1049.47500000 * (1/260)
                        = 4.036442308



Total Commission (in BNB) = Standard commission (Discounted) + Tax commission (in BNB) + Special commission (in BNB)
                          = 0.000010091 + 0.00015473 + 4.036442308
                          = 4.036607129


Si vous n'avez pas suffisamment de BNB pour payer la commission réduite, la commission complète sera prélevée sur le montant USDT que vous avez reçu.

FAQ sur les ordres stop suiveurs au comptant
Qu'est-ce qu'un ordre stop suiveur
Le stop suiveur est un type d'ordre conditionnel dont le prix de déclenchement est dynamique et influencé par les variations de prix du marché. Pour l'API SPOT, le changement requis pour déclencher la saisie de l'ordre est spécifié dans le trailingDeltaparamètre et défini en BIPS.

Intuitivement, les ordres stop suiveurs permettent un mouvement de prix illimité dans une direction bénéfique pour l'ordre et un mouvement limité dans une direction préjudiciable.

Ordres d'achat : les prix bas sont une bonne chose. Des baisses de prix illimitées sont autorisées, mais l'ordre sera déclenché après une augmentation du prix du delta fourni, par rapport au prix de transaction le plus bas depuis sa soumission.

Ordres de vente : des prix élevés sont une bonne chose. Des hausses de prix illimitées sont autorisées, mais l'ordre sera déclenché après une baisse du prix correspondant au delta fourni, par rapport au prix de transaction le plus élevé depuis sa soumission.

Que sont les BIP
Les points de base, également appelés BIP ou BIPS, sont utilisés pour indiquer une variation en pourcentage.

Référence de conversion BIPS :

BIPS	Pourcentage	Multiplicateur
1	0,01%	0,0001
10	0,1%	0,001
100	1%	0,01
1000	10%	0,1
Par exemple, un STOP_LOSS SELLordre avec un a trailingDeltade 100 est un ordre stop suiveur qui sera déclenché après une baisse de prix de 1 % par rapport au prix le plus élevé après avoir passé l'ordre.

Quels types d'ordres peuvent être des ordres stop suiveurs
Les ordres stop suiveurs sont pris en charge pour les ordres contingents tels que STOP_LOSS, STOP_LOSS_LIMIT, TAKE_PROFITet TAKE_PROFIT_LIMIT.

Les ordres OCO prennent également en charge les ordres stop suiveurs dans la branche contingente. Dans ce cas, si la condition stop suiveur est déclenchée, la branche limite de l'ordre OCO sera annulée.

Comment placer un ordre stop suiveur
Les ordres stop suiveurs sont saisis de la même manière que les ordres STOP_LOSS, STOP_LOSS_LIMIT, TAKE_PROFITou classiques TAKE_PROFIT_LIMIT, mais avec un paramètre supplémentaire trailingDelta. Ce paramètre doit être compris dans la plage du TRAILING_DELTAfiltre pour ce symbole.

Contrairement aux ordres conditionnels classiques, ce stopPriceparamètre est facultatif pour les ordres stop suiveurs. S'il est fourni, l'ordre ne commencera à suivre les variations de prix qu'une fois la stopPricecondition remplie. S'il stopPriceest omis, l'ordre commencera à suivre les variations de prix à partir de la transaction suivante.

Quels types de changements de prix déclencheront mon ordre stop suiveur
Type d'ordre suiveur	Côté	Condition de prix stop	Mouvement des prix du marché requis pour déclencher
TAKE_PROFIT	VENDRE	prix du marché >= prix stop	diminution du maximum
TAKE_PROFIT_LIMIT	VENDRE	prix du marché >= prix stop	diminution du maximum
STOP_LOSS	VENDRE	prix du marché <= prix stop	diminution du maximum
STOP_LOSS_LIMIT	VENDRE	prix du marché <= prix stop	diminution du maximum
STOP_LOSS	ACHETER	prix du marché >= prix stop	augmentation du minimum
STOP_LOSS_LIMIT	ACHETER	prix du marché >= prix stop	augmentation du minimum
TAKE_PROFIT	ACHETER	prix du marché <= prix stop	augmentation du minimum
TAKE_PROFIT_LIMIT	ACHETER	prix du marché <= prix stop	augmentation du minimum
Comment passer le TRAILING_DELTAfiltre 
Pour les commandes STOP_LOSS BUY, STOP_LOSS_LIMIT BUY, TAKE_PROFIT SELL, et TAKE_PROFIT_LIMIT SELL:

trailingDelta>=minTrailingAboveDelta
trailingDelta<=maxTrailingAboveDelta
Pour les commandes STOP_LOSS SELL, STOP_LOSS_LIMIT SELL, TAKE_PROFIT BUY, et TAKE_PROFIT_LIMIT BUY:

trailingDelta>=minTrailingBelowDelta
trailingDelta<=maxTrailingBelowDelta
 d'ordres stop suiveurs
Scénario A - 
Une 12:01:00transaction est effectuée au prix de 40 000 $ et un STOP_LOSS_LIMITordre est passé sur BUYle marché boursier. Cet ordre comporte un seuil stopPricede 44 000 $, un trailingDeltaseuil de 500 $ (5 %) et une limite pricede 45 000 $.

Entre 12:01:00et, 12:02:00une série de transactions linéaires a entraîné une baisse du dernier cours, qui a atteint 37 000. Il s'agit d'une baisse de 7,5 %, soit 750 BIPS, bien supérieure à celle de l'ordre trailingDelta. Cependant, comme l'ordre n'a pas encore commencé à suivre le cours, son évolution est ignorée et l'ordre reste conditionnel.

Entre 12:02:00et, 12:03:00une série de transactions linéaires entraîne une hausse du dernier cours. Lorsqu'une transaction atteint ou dépasse ce niveau, l' stopPriceordre commence immédiatement à suivre les variations de prix ; la première transaction qui remplit cette condition définit le « prix le plus bas ». Dans ce cas, le prix le plus bas est de 44 000 $. Si la hausse est de 500 BIPS à partir de 44 000 $, l'ordre est déclenché. La série de transactions linéaires continue d'augmenter jusqu'au dernier cours, pour se terminer à 45 000 $.

Entre 12:03:00et, 12:04:00une série de transactions linéaires a entraîné une hausse du dernier prix, qui a atteint 46 000. Il s'agit d'une hausse d'environ 454 BIPS par rapport au prix le plus bas précédemment enregistré, mais elle n'est pas suffisante pour déclencher l'ordre.

Entre 12:04:00et, 12:05:00une série de transactions linéaires entraîne une baisse du dernier cours, qui atteint 42 000. Il s'agit d'une baisse par rapport au prix le plus bas précédemment enregistré. Si la hausse est de 500 BIPS par rapport à 42 000, l'ordre sera déclenché.

Entre 12:05:00et, 12:05:30une série de transactions linéaires entraîne une hausse du dernier prix à 44 100. Cette transaction est égale ou supérieure à l'exigence de l'ordre de 500 BIPS, car 44,100 = 42,000 * 1.05. Cela déclenche l'ordre et commence à fonctionner contre le carnet d'ordres à son prix limite de 45 000.

image
Scénario B - 
Une transaction est 12:01:00effectuée au prix de 40 000 $ et un STOP_LOSS_LIMITordre est passé sur SELLle marché boursier. Cet ordre comporte un seuil stopPricede 39 000 $, un trailingDeltaseuil de 1 000 $ (10 %) et une limite pricede 38 000 $.

Entre 12:01:00et 12:02:00une série de transactions linéaires ont conduit à une augmentation du dernier prix, se terminant à 41 500.

Entre 12:02:00et, 12:03:00une série de transactions linéaires entraîne une baisse du dernier cours. Lorsqu'une transaction atteint ou dépasse ce niveau, l' stopPriceordre commence immédiatement à suivre les variations de prix ; la première transaction qui remplit cette condition définit le « prix le plus élevé ». Dans ce cas, le prix le plus élevé est de 39 000 BIPS et, en cas de baisse de 1 000 BIPS par rapport à 39 000, l'ordre est déclenché.

Entre 12:03:00et, 12:04:00une série de transactions linéaires a entraîné une baisse du dernier prix, qui a atteint 37 000. Il s'agit d'une baisse d'environ 512 BIPS par rapport au prix le plus élevé précédemment enregistré, mais elle n'est pas suffisante pour déclencher l'ordre.

Entre 12:04:00et, 12:05:00une série de transactions linéaires entraîne une hausse du dernier prix, qui atteint 41 000. Il s'agit d'une hausse par rapport au prix le plus élevé précédemment enregistré. En cas de baisse de 1 000 BIPS par rapport à 41 000, l'ordre est déclenché.

Entre 12:05:00et, 12:05:30une série de transactions linéaires entraîne une baisse du dernier prix à 36 900. Cette transaction est égale ou supérieure à l'exigence de l'ordre de 1 000 BIPS, car 36,900 = 41,000 * 0.90. Cela déclenche l'ordre et commence à fonctionner contre le carnet d'ordres à son prix limite de 38 000.

image
Scénario C - 
Une 12:01:00transaction est effectuée au prix de 40 000 $ et un TAKE_PROFIT_LIMITordre est passé sur BUYle marché boursier. Cet ordre comporte un seuil stopPricede 38 000 $, un trailingDeltaseuil de 850 $ (8,5 %) et une limite pricede 38 500 $.

Entre 12:01:00et 12:02:00une série de transactions linéaires ont conduit à une augmentation du dernier prix, se terminant à 42 000.

Entre 12:02:00et, 12:03:00une série de transactions linéaires entraîne une baisse du dernier cours. Lorsqu'une transaction atteint ou dépasse ce niveau, l' stopPriceordre commence immédiatement à suivre les variations de prix ; la première transaction qui remplit cette condition définit le « prix le plus bas ». Dans ce cas, le prix le plus bas est de 38 000 ; si une hausse de 850 BIPS est observée à partir de 38 000, l'ordre est déclenché.

La série de transactions linéaires continue de baisser jusqu'au dernier prix, pour se terminer à 37 000. Si le cours augmente de 850 BIPS à partir de 37 000, l'ordre sera déclenché.

Entre 12:03:00et, 12:04:00une série de transactions linéaires a entraîné une hausse du dernier prix, qui a atteint 39 000. Il s'agit d'une hausse d'environ 540 BIPS par rapport au prix le plus bas précédemment enregistré, mais elle n'est pas suffisante pour déclencher l'ordre.

Entre 12:04:00et, 12:05:00une série de transactions linéaires entraîne une baisse du dernier prix, qui atteint 38 000. Ce dernier ne dépasse pas le prix le plus bas précédemment enregistré, ce qui ne modifie pas le prix de déclenchement de l'ordre.

Entre 12:05:00et, 12:05:30une série de transactions linéaires a entraîné une hausse du dernier prix à 40 145. Cette transaction est égale ou supérieure à l'exigence de l'ordre de 850 BIPS, car 40,145 = 37,000 * 1.085. Cela déclenche l'ordre et commence à fonctionner contre le carnet d'ordres à son prix limite de 38 500.

image
Scénario D - 
Une 12:01:00transaction est effectuée au prix de 40 000 $ et un TAKE_PROFIT_LIMITordre est passé sur SELLle marché boursier. Cet ordre comporte un seuil stopPricede 42 000 $, un trailingDeltaseuil de 750 $ (7,5 %) et une limite pricede 41 000 $.

Entre 12:01:00et 12:02:00une série de transactions linéaires ont conduit à une augmentation du dernier prix, se terminant à 41 500.

Entre 12:02:00et 12:03:00une série de transactions linéaires ont conduit à une baisse du dernier prix, se terminant à 39 000.

Entre 12:03:00et, 12:04:00une série de transactions linéaires entraîne une hausse du dernier cours. Lorsqu'une transaction atteint ou dépasse ce niveau, l' stopPriceordre commence immédiatement à suivre les variations de prix ; la première transaction qui remplit cette condition définit le « prix le plus élevé ». Dans ce cas, le prix le plus élevé est de 42 000 ; si la baisse est de 750 BIPS par rapport à 42 000, l'ordre est déclenché.

La série de transactions linéaires continue d'augmenter jusqu'au dernier prix, pour atteindre 45 000. En cas de baisse de 750 BIPS par rapport à 45 000, l'ordre sera déclenché.

Entre 12:04:00et, 12:05:00une série de transactions linéaires a entraîné une baisse du dernier cours, qui a atteint 44 000. Il s'agit d'une baisse d'environ 222 BIPS par rapport au prix le plus élevé précédemment enregistré, mais elle n'est pas suffisante pour déclencher l'ordre.

Entre 12:05:00et, 12:06:00une série de transactions linéaires entraîne une hausse du dernier prix, qui atteint 46 500. Il s'agit d'une hausse par rapport au prix le plus élevé précédemment enregistré. En cas de baisse de 750 BIPS par rapport à 46 500, l'ordre est déclenché.

Entre 12:06:00et, 12:06:50une série de transactions linéaires a entraîné une baisse du dernier cours à 43 012,5. Cette transaction est égale ou supérieure à l'exigence de l'ordre de 750 BIPS, car 43,012.5 = 46,500 * 0.925. Cela déclenche l'ordre et commence à fonctionner contre le carnet d'ordres à son prix limite de 41 000.

image
Scénario E - Ordre stop suiveur sans 
Une 12:01:00transaction est effectuée au prix de 40 000 BIPS et un STOP_LOSS_LIMITordre est placé sur la SELLplateforme d'échange. Cet ordre a un seuil trailingDeltade 700 BIPS (7 %), une limite pricede 39 000 BIPS et aucun BIPS stopPrice. Une fois placé, l'ordre commence à suivre les variations de prix. Si le prix de 40 000 BIPS chute de 700 BIPS, l'ordre est déclenché.

Entre 12:01:00et, 12:02:00une série de transactions linéaires entraîne une hausse du dernier prix, qui atteint 42 000. Il s'agit d'une hausse par rapport au prix le plus élevé précédemment enregistré. En cas de baisse de 700 BIPS par rapport à 42 000, l'ordre est déclenché.

Entre 12:02:00et, 12:03:00une série de transactions linéaires a entraîné une baisse du dernier cours, qui a atteint 39 500. Il s'agit d'une baisse d'environ 595 BIPS par rapport au prix le plus élevé précédemment enregistré, mais elle n'est pas suffisante pour déclencher l'ordre.

Entre 12:03:00et, 12:04:00une série de transactions linéaires entraîne une hausse du dernier prix, qui atteint 45 500. Il s'agit d'une hausse par rapport au prix le plus élevé précédemment enregistré. En cas de baisse de 700 BIPS par rapport à 45 500, l'ordre est déclenché.

Entre 12:04:00et, 12:04:45une série de transactions linéaires a entraîné une baisse du dernier cours à 42 315. Cette transaction est égale ou supérieure à l'exigence de l'ordre de 700 BIPS, car 42,315 = 45,500 * 0.93. Cela déclenche l'ordre et commence à fonctionner contre le carnet d'ordres à son prix limite de 39 000.

image
 d'ordres stop suiveurs
En supposant un dernier prix de 40 000.

Placement d'un ordre stop suiveur STOP_LOSS_LIMIT BUY, avec un prix de 42 000,0 et un stop suiveur de 5 %.

# Excluding stop price
POST 'https://api.binance.com/api/v3/order?symbol=BTCUSDT&side=BUY&type=STOP_LOSS_LIMIT&timeInForce=GTC&quantity=0.01&price=42000&trailingDelta=500&timestamp=<timestamp>&signature=<signature>'

# Including stop price of 43,000
POST 'https://api.binance.com/api/v3/order?symbol=BTCUSDT&side=BUY&type=STOP_LOSS_LIMIT&timeInForce=GTC&quantity=0.01&price=42000&stopPrice=43000&trailingDelta=500&timestamp=<timestamp>&signature=<signature>'


Placement d'un ordre stop suiveur STOP_LOSS_LIMIT SELL, avec un prix de 37 500,0 et un stop suiveur de 2,5 %.

# Excluding stop price
POST 'https://api.binance.com/api/v3/order?symbol=BTCUSDT&side=SELL&type=STOP_LOSS_LIMIT&timeInForce=GTC&quantity=0.01&price=37500&trailingDelta=250&timestamp=<timestamp>&signature=<signature>'

# Including stop price of 39,000
POST 'https://api.binance.com/api/v3/order?symbol=BTCUSDT&side=SELL&type=STOP_LOSS_LIMIT&timeInForce=GTC&quantity=0.01&price=37500&stopPrice=39000&trailingDelta=250&timestamp=<timestamp>&signature=<signature>'


Placement d'un ordre stop suiveur TAKE_PROFIT_LIMIT BUY, avec un prix de 38 000,0 et un stop suiveur de 5 %.

# Excluding stop price
POST 'https://api.binance.com/api/v3/order?symbol=BTCUSDT&side=BUY&type=TAKE_PROFIT_LIMIT&timeInForce=GTC&quantity=0.01&price=38000&trailingDelta=500&timestamp=<timestamp>&signature=<signature>'

# Including stop price of 36,000
POST 'https://api.binance.com/api/v3/order?symbol=BTCUSDT&side=BUY&type=TAKE_PROFIT_LIMIT&timeInForce=GTC&quantity=0.01&price=38000&stopPrice=36000&trailingDelta=500&timestamp=<timestamp>&signature=<signature>'


Placement d'un ordre stop suiveur TAKE_PROFIT_LIMIT SELL, avec un prix de 41 500,0 et un stop suiveur de 1,75 %.

# Excluding stop price
POST 'https://api.binance.com/api/v3/order?symbol=BTCUSDT&side=SELL&type=TAKE_PROFIT_LIMIT&timeInForce=GTC&quantity=0.01&price=41500&trailingDelta=175&timestamp=<timestamp>&signature=<signature>'

# Including stop price of 42,500
POST 'https://api.binance.com/api/v3/order?symbol=BTCUSDT&side=SELL&type=TAKE_PROFIT_LIMIT&timeInForce=GTC&quantity=0.01&price=41500&stopPrice=42500&trailingDelta=175&timestamp=<timestamp>&signature=<signature>'




FAQ sur la prévention du commerce libre-service (STP)
Clause de non-responsabilité:

Les commissions et les prix utilisés ici sont fictifs et n'impliquent rien sur la configuration réelle de l'échange en direct.
Qu'est-ce que la prévention du commerce libre-service
La prévention de l'auto-échange (ou STP) empêche les commandes des utilisateurs, ou celles de l'utilisateur, tradeGroupIdde correspondre aux leurs.

Qu'est-ce qui définit un échange personnel 
Un échange personnel peut se produire dans les deux cas :

L'ordre a été négocié sur le même compte.
L'ordre a été négocié sur un compte avec le même tradeGroupId.
Que se passe-t-il lorsque le STP est déclenché
Il existe cinq modes possibles pour ce que le système fait lorsqu'un ordre crée un auto-échange.

NONE- Ce mode exempte l'ordre de la prévention d'auto-échange. Les identifiants de comptes ou de groupes d'échange ne seront pas comparés, aucun ordre n'expirera et l'échange aura lieu.

EXPIRE_TAKER- Ce mode empêche une transaction en expirant immédiatement la quantité restante de l'ordre du preneur.

EXPIRE_MAKER- Ce mode empêche une transaction en expirant immédiatement la quantité restante de l'ordre du fabricant potentiel.

EXPIRE_BOTH- Ce mode empêche une transaction en expirant immédiatement les quantités restantes des ordres du preneur et du créateur potentiel.

DECREMENT Ce mode augmente la valeur prevented quantitydes deux commandes du montant de la correspondance empêchée. La plus petite des deux commandes expirera, ou les deux si elles ont la même quantité.

L'événement STP se produit en fonction du mode STP de l' ordre du preneur .
Ainsi, le mode STP d'un ordre enregistré dans le carnet d'ordres n'est plus pertinent et sera ignoré pour tout traitement ultérieur.

Qu'est-ce qu'un identifiant de groupe commercial
Différents comptes ayant le même compte tradeGroupIdsont considérés comme faisant partie du même groupe commercial. Les ordres soumis par les membres d'un même groupe sont éligibles au STP selon le mode STP du preneur d'ordre.

Un utilisateur peut confirmer si ses comptes sont sous le même nom tradeGroupIdà partir de l'API GET /api/v3/account(API REST) ou account.status(API WebSocket) pour chaque compte.

Le champ est également présent dans la réponse pour GET /api/v3/preventedMatches(API REST) ou myPreventedMatches(API WebSocket).

Si la valeur est -1, alors le tradeGroupIdn'a pas été défini pour ce compte, donc le STP ne peut avoir lieu qu'entre les commandes du même compte.

Qu'est-ce qu'un match empêché
Lorsqu'une auto-transaction est empêchée, une correspondance empêchée est créée. Les ordres de cette correspondance voient leurs quantités empêchées augmentées et un ou plusieurs ordres expirent.

Il ne faut pas confondre cela avec une transaction, car aucune commande ne correspondra.

Il s’agit d’un enregistrement des ordres qui auraient pu être négociés automatiquement.

Cela peut être interrogé via le point de terminaison GET /api/v3/preventedMatchessur l'API REST ou myPreventedMatchessur l'API WebSocket.

Voici un exemple de demande de sortie pour référence :

[
  {
    "symbol": "BTCDUSDT",                       //Symbol of the orders
    "preventedMatchId": 8,                      //Identifies the prevented match of the expired order(s) for the symbol.
    "takerOrderId": 12,                         //Order Id of the Taker Order
    "makerOrderId": 10,                         //Order Id of the Maker Order
    "tradeGroupId": 1,                          //Identifies the Trade Group Id. (If the account is not part of a trade group, this will be -1.)
    "selfTradePreventionMode": "EXPIRE_BOTH",   //STP mode that expired the order(s).
    "price": "50.00000000",                     //Price at which the match occurred.
    "takerPreventedQuantity": "1.00000000",     //Taker's remaining quantity before the STP. Only appears if the STP mode is EXPIRE_TAKER, EXPIRE_BOTH or DECREMENT.
    "makerPreventedQuantity": "10.00000000",    //Maker's remaining quantity before the STP. Only appears if the STP mode is EXPIRE_MAKER, EXPIRE_BOTH, or DECREMENT.
    "transactTime": 1663190634060               //Time the order(s) expired due to STP.
  }
]


Qu'est-ce que la « quantité empêchée 
Les événements STP expirent la quantité des ordres ouverts. Les modes STP EXPIRE_TAKER, EXPIRE_MAKERet EXPIRE_BOTHexpirent toute la quantité restante des ordres concernés, ce qui entraîne l'expiration de l'ordre ouvert dans son intégralité.

La quantité empêchée correspond à la quantité expirée suite à des événements STP pour une commande donnée. Les rapports d'exécution des flux utilisateur pour les commandes concernées par STP peuvent contenir les champs suivants :

{
  "A":"3.000000", // Prevented Quantity
  "B":"3.000000"  // Last Prevented Quantity
}

Best présent pour le type d'exécution TRADE_PREVENTIONet correspond à la quantité expirée en raison de cet événement STP individuel.

Acorrespond à la quantité cumulée expirée en raison d'un STP sur la durée de vie de la commande. Pour les modes EXPIRE_TAKER, EXPIRE_MAKERet , EXPIRE_BOTHcette valeur sera toujours identique à B.

Les réponses API pour les commandes expirées en raison du STP auront également un preventedQuantitychamp indiquant la quantité cumulée expirée en raison du STP sur la durée de vie de la commande.

Lorsqu'une commande est ouverte, l'équation suivante est vraie :

original order quantity - executed quantity - prevented quantity = quantity available for further execution


Lorsque la quantité disponible d'une commande tombe à zéro, la commande est supprimée du carnet de commandes et le statut devient EXPIRED_IN_MATCH, FILLED, ou EXPIRED.

Comment savoir quel symbole utilise STP
Les symboles peuvent être configurés pour autoriser différents ensembles de modes STP et prendre différents modes STP par défaut.

defaultSelfTradePreventionMode- Les commandes utiliseront ce mode STP si l'utilisateur n'en fournit pas lors de la passation de la commande.

allowedSelfTradePreventionModes- Définit l'ensemble autorisé de modes STP pour le placement d'ordres sur ce symbole.

Par exemple, si un symbole a la configuration suivante :

"defaultSelfTradePreventionMode": "NONE",
"allowedSelfTradePreventionModes": [
    "NONE",
    "EXPIRE_TAKER",
    "EXPIRE_BOTH"
  ]

Cela signifie donc que si un utilisateur envoie une commande sans selfTradePreventionModefournir de valeur, la commande envoyée aura la valeur de NONE.

Si un utilisateur souhaite spécifier explicitement le mode, il peut transmettre l'énumération NONE, EXPIRE_TAKER, ou EXPIRE_BOTH.

Si un utilisateur tente de spécifier EXPIRE_MAKERdes commandes sur ce symbole, il recevra une erreur :

{
    "code": -1013,
    "msg": "This symbol does not allow the specified self-trade prevention mode."
}


Comment savoir si une commande a expiré en raison d'un STP
La commande aura le statut EXPIRED_IN_MATCH.

 STP
Pour tous ces cas, supposons que toutes les commandes pour ces exemples sont effectuées sur le même compte.

Scénario A - Un utilisateur envoie une nouvelle commande avec selfTradePreventionMode : NONEqui correspondra à une autre de ses commandes qui est déjà dans le carnet.

Maker Order: symbol=BTCUSDT side=BUY  type=LIMIT quantity=1 price=1 selfTradePreventionMode=NONE
Taker Order: symbol=BTCUSDT side=SELL type=LIMIT quantity=1 price=1 selfTradePreventionMode=NONE


Résultat : aucun STP n'est déclenché et les commandes correspondront.

Statut de la commande du fabricant

{
  "symbol": "BTCUSDT",
  "orderId": 2,
  "orderListId": -1,
  "clientOrderId": "FaDk4LPRxastaICEFE9YTf",
  "price": "1.000000",
  "origQty": "1.000000",
  "executedQty": "1.000000",
  "cummulativeQuoteQty": "1.000000",
  "status": "FILLED",
  "timeInForce": "GTC",
  "type": "LIMIT",
  "side": "BUY",
  "stopPrice": "0.000000",
  "icebergQty": "0.000000",
  "time": 1670217090310,
  "updateTime": 1670217090330,
  "isWorking": true,
  "workingTime": 1670217090310,
  "origQuoteOrderQty": "0.000000",
  "selfTradePreventionMode": "NONE"
}

Statut de la commande du preneur

{
  "symbol": "BTCUSDT",
  "orderId": 3,
  "orderListId": -1,
  "clientOrderId": "Ay48Vtpghnsvy6w8RPQEde",
  "transactTime": 1670207731263,
  "price": "1.000000",
  "origQty": "1.000000",
  "executedQty": "1.000000",
  "cummulativeQuoteQty": "1.000000",
  "status": "FILLED",
  "timeInForce": "GTC",
  "type": "LIMIT",
  "side": "SELL",
  "workingTime": 1670207731263,
  "fills": [
    {
      "price": "1.000000",
      "qty": "1.000000",
      "commission": "0.000000",
      "commissionAsset": "USDT",
      "tradeId": 1
    }
  ],
  "selfTradePreventionMode": "NONE"
}

Scénario B - Un utilisateur envoie une commande EXPIRE_MAKERqui correspondrait à ses commandes déjà présentes dans le carnet.

Maker Order 1: symbol=BTCUSDT side=BUY  type=LIMIT quantity=1.2 price=1.2 selfTradePreventionMode=NONE
Maker Order 2: symbol=BTCUSDT side=BUY  type=LIMIT quantity=1.3 price=1.1 selfTradePreventionMode=NONE
Maker Order 3: symbol=BTCUSDT side=BUY  type=LIMIT quantity=8.1 price=1   selfTradePreventionMode=NONE
Taker Order 1: symbol=BTCUSDT side=SELL type=LIMIT quantity=3   price=1   selfTradePreventionMode=EXPIRE_MAKER


Résultat : Les ordres qui étaient sur le carnet expireront en raison du STP, et l'ordre du preneur sera ajouté au carnet.

Commande du fabricant 1

{
  "symbol": "BTCUSDT",
  "orderId": 2,
  "orderListId": -1,
  "clientOrderId": "wpNzhSclc16pV8g5THIOR3",
  "price": "1.200000",
  "origQty": "1.200000",
  "executedQty": "0.000000",
  "cummulativeQuoteQty": "0.000000",
  "status": "EXPIRED_IN_MATCH",
  "timeInForce": "GTC",
  "type": "LIMIT",
  "side": "BUY",
  "stopPrice": "0.000000",
  "icebergQty": "0.000000",
  "time": 1670217957437,
  "updateTime": 1670217957498,
  "isWorking": true,
  "workingTime": 1670217957437,
  "origQuoteOrderQty": "0.000000",
  "selfTradePreventionMode": "NONE",
  "preventedMatchId": 0,
  "preventedQuantity": "1.200000"
}

Commande du fabricant 2

{
  "symbol": "BTCUSDT",
  "orderId": 3,
  "orderListId": -1,
  "clientOrderId": "ZT9emqia99V7x8B6FW0pFF",
  "price": "1.100000",
  "origQty": "1.300000",
  "executedQty": "0.000000",
  "cummulativeQuoteQty": "0.000000",
  "status": "EXPIRED_IN_MATCH",
  "timeInForce": "GTC",
  "type": "LIMIT",
  "side": "BUY",
  "stopPrice": "0.000000",
  "icebergQty": "0.000000",
  "time": 1670217957458,
  "updateTime": 1670217957498,
  "isWorking": true,
  "workingTime": 1670217957458,
  "origQuoteOrderQty": "0.000000",
  "selfTradePreventionMode": "NONE",
  "preventedMatchId": 1,
  "preventedQuantity": "1.300000"
}

Commande du fabricant 3

{
  "symbol": "BTCUSDT",
  "orderId": 4,
  "orderListId": -1,
  "clientOrderId": "8QZ3taGcU4gND59TxHAcR0",
  "price": "1.000000",
  "origQty": "8.100000",
  "executedQty": "0.000000",
  "cummulativeQuoteQty": "0.000000",
  "status": "EXPIRED_IN_MATCH",
  "timeInForce": "GTC",
  "type": "LIMIT",
  "side": "BUY",
  "stopPrice": "0.000000",
  "icebergQty": "0.000000",
  "time": 1670217957478,
  "updateTime": 1670217957498,
  "isWorking": true,
  "workingTime": 1670217957478,
  "origQuoteOrderQty": "0.000000",
  "selfTradePreventionMode": "NONE",
  "preventedMatchId": 2,
  "preventedQuantity": "8.100000"
}

Sortie de l'ordre du preneur

{
  "symbol": "BTCUSDT",
  "orderId": 5,
  "orderListId": -1,
  "clientOrderId": "WRzbhp257NhZsIJW4y2Nri",
  "transactTime": 1670217957498,
  "price": "1.000000",
  "origQty": "3.000000",
  "executedQty": "0.000000",
  "cummulativeQuoteQty": "0.000000",
  "status": "NEW",
  "timeInForce": "GTC",
  "type": "LIMIT",
  "side": "SELL",
  "workingTime": 1670217957498,
  "fills": [],
  "preventedMatches": [
    {
      "preventedMatchId": 0,
      "makerOrderId": 2,
      "price": "1.200000",
      "makerPreventedQuantity": "1.200000"
    },
    {
      "preventedMatchId": 1,
      "makerOrderId": 3,
      "price": "1.100000",
      "makerPreventedQuantity": "1.300000"
    },
    {
      "preventedMatchId": 2,
      "makerOrderId": 4,
      "price": "1.000000",
      "makerPreventedQuantity": "8.100000"
    }
  ],
  "selfTradePreventionMode": "EXPIRE_MAKER"
}

Scénario C - Un utilisateur envoie une commande EXPIRE_TAKERqui correspondrait à ses commandes déjà présentes dans le carnet.

Maker Order 1: symbol=BTCUSDT side=BUY  type=LIMIT quantity=1.2 price=1.2  selfTradePreventionMode=NONE
Maker Order 2: symbol=BTCUSDT side=BUY  type=LIMIT quantity=1.3 price=1.1  selfTradePreventionMode=NONE
Maker Order 3: symbol=BTCUSDT side=BUY  type=LIMIT quantity=8.1 price=1    selfTradePreventionMode=NONE
Taker Order 1: symbol=BTCUSDT side=SELL type=LIMIT quantity=3   price=1    selfTradePreventionMode=EXPIRE_TAKER


Résultat : Les ordres déjà présents dans le carnet resteront, tandis que l'ordre du preneur expirera.

Commande du fabricant 1

{
  "symbol": "BTCUSDT",
  "orderId": 2,
  "orderListId": -1,
  "clientOrderId": "NpwW2t0L4AGQnCDeNjHIga",
  "price": "1.200000",
  "origQty": "1.200000",
  "executedQty": "0.000000",
  "cummulativeQuoteQty": "0.000000",
  "status": "NEW",
  "timeInForce": "GTC",
  "type": "LIMIT",
  "side": "BUY",
  "stopPrice": "0.000000",
  "icebergQty": "0.000000",
  "time": 1670219811986,
  "updateTime": 1670219811986,
  "isWorking": true,
  "workingTime": 1670219811986,
  "origQuoteOrderQty": "0.000000",
  "selfTradePreventionMode": "NONE"
}

Commande du fabricant 2

{
  "symbol": "BTCUSDT",
  "orderId": 3,
  "orderListId": -1,
  "clientOrderId": "TSAmJqGWk4YTB2yA9p04UO",
  "price": "1.100000",
  "origQty": "1.300000",
  "executedQty": "0.000000",
  "cummulativeQuoteQty": "0.000000",
  "status": "NEW",
  "timeInForce": "GTC",
  "type": "LIMIT",
  "side": "BUY",
  "stopPrice": "0.000000",
  "icebergQty": "0.000000",
  "time": 1670219812007,
  "updateTime": 1670219812007,
  "isWorking": true,
  "workingTime": 1670219812007,
  "origQuoteOrderQty": "0.000000",
  "selfTradePreventionMode": "NONE"
}

Commande du fabricant 3

{
  "symbol": "BTCUSDT",
  "orderId": 4,
  "orderListId": -1,
  "clientOrderId": "L6FmpCJJP6q4hCNv4MuZDG",
  "price": "1.000000",
  "origQty": "8.100000",
  "executedQty": "0.000000",
  "cummulativeQuoteQty": "0.000000",
  "status": "NEW",
  "timeInForce": "GTC",
  "type": "LIMIT",
  "side": "BUY",
  "stopPrice": "0.000000",
  "icebergQty": "0.000000",
  "time": 1670219812026,
  "updateTime": 1670219812026,
  "isWorking": true,
  "workingTime": 1670219812026,
  "origQuoteOrderQty": "0.000000",
  "selfTradePreventionMode": "NONE"
}

Sortie de l'ordre Taker

{
  "symbol": "BTCUSDT",
  "orderId": 5,
  "orderListId": -1,
  "clientOrderId": "kocvDAi4GNN2y1l1Ojg1Ri",
  "price": "1.000000",
  "origQty": "3.000000",
  "executedQty": "0.000000",
  "cummulativeQuoteQty": "0.000000",
  "status": "EXPIRED_IN_MATCH",
  "timeInForce": "GTC",
  "type": "LIMIT",
  "side": "SELL",
  "stopPrice": "0.000000",
  "icebergQty": "0.000000",
  "time": 1670219812046,
  "updateTime": 1670219812046,
  "isWorking": true,
  "workingTime": 1670219812046,
  "origQuoteOrderQty": "0.000000",
  "selfTradePreventionMode": "EXPIRE_TAKER",
  "preventedMatchId": 0,
  "preventedQuantity": "3.000000"
}

Scénario D - Un utilisateur a une commande sur le livre, puis envoie une commande EXPIRE_BOTHqui correspondrait à la commande existante.

Maker Order: symbol=BTCUSDT side=BUY  type=LIMIT quantity=1 price=1 selfTradePreventionMode=NONE
Taker Order: symbol=BTCUSDT side=SELL type=LIMIT quantity=3 price=1 selfTradePreventionMode=EXPIRE_BOTH


Résultat : les deux commandes expireront.

Commande du fabricant

{
  "symbol": "BTCUSDT",
  "orderId": 2,
  "orderListId": -1,
  "clientOrderId": "2JPC8xjpLq6Q0665uYWAcs",
  "price": "1.000000",
  "origQty": "1.000000",
  "executedQty": "0.000000",
  "cummulativeQuoteQty": "0.000000",
  "status": "EXPIRED_IN_MATCH",
  "timeInForce": "GTC",
  "type": "LIMIT",
  "side": "BUY",
  "stopPrice": "0.000000",
  "icebergQty": "0.000000",
  "time": 1673842412831,
  "updateTime": 1673842413170,
  "isWorking": true,
  "workingTime": 1673842412831,
  "origQuoteOrderQty": "0.000000",
  "selfTradePreventionMode": "NONE",
  "preventedMatchId": 0,
  "preventedQuantity": "1.000000"
}

Ordre du preneur

{
  "symbol": "BTCUSDT",
  "orderId": 5,
  "orderListId": -1,
  "clientOrderId": "qMaz8yrOXk2iUIz74cFkiZ",
  "transactTime": 1673842413170,
  "price": "1.000000",
  "origQty": "3.000000",
  "executedQty": "0.000000",
  "cummulativeQuoteQty": "0.000000",
  "status": "EXPIRED_IN_MATCH",
  "timeInForce": "GTC",
  "type": "LIMIT",
  "side": "SELL",
  "workingTime": 1673842413170,
  "fills": [],
  "preventedMatches": [
    {
      "preventedMatchId": 0,
      "makerOrderId": 2,
      "price": "1.000000",
      "takerPreventedQuantity": "3.000000",
      "makerPreventedQuantity": "1.000000"
    }
  ],
  "selfTradePreventionMode": "EXPIRE_BOTH",
  "tradeGroupId": 1,
  "preventedQuantity": "3.000000"
}

Scénario E - Un utilisateur a une commande sur le carnet avec EXPIRE_MAKER, puis envoie une nouvelle commande avec EXPIRE_TAKERqui correspondrait à la commande existante.

Maker Order: symbol=BTCUSDT side=BUY  type=LIMIT quantity=1 price=1 selfTradePreventionMode=EXPIRE_MAKER
Taker Order: symbol=BTCUSDT side=SELL type=LIMIT quantity=1 price=1 selfTradePreventionMode=EXPIRE_TAKER


Résultat : Le mode STP de l'ordre du preneur sera utilisé, donc l'ordre du preneur sera expiré.

Commande du fabricant

{
    "symbol": "BTCUSDT",
    "orderId": 0,
    "orderListId": -1,
    "clientOrderId": "jFUap8iFwwgqIpOfAL60GS",
    "price": "1.000000",
    "origQty": "1.000000",
    "executedQty": "0.000000",
    "cummulativeQuoteQty": "0.000000",
    "status": "NEW",
    "timeInForce": "GTC",
    "type": "LIMIT",
    "side": "BUY",
    "stopPrice": "0.000000",
    "icebergQty": "0.000000",
    "time": 1670220769261,
    "updateTime": 1670220769261,
    "isWorking": true,
    "workingTime": 1670220769261,
    "origQuoteOrderQty": "0.000000",
    "selfTradePreventionMode": "EXPIRE_MAKER"
}

Ordre du preneur

{
    "symbol": "BTCUSDT",
    "orderId": 1,
    "orderListId": -1,
    "clientOrderId": "zxrvnNNm1RXC3rkPLUPrc1",
    "transactTime": 1670220800315,
    "price": "1.000000",
    "origQty": "1.000000",
    "executedQty": "0.000000",
    "cummulativeQuoteQty": "0.000000",
    "status": "EXPIRED_IN_MATCH",
    "timeInForce": "GTC",
    "type": "LIMIT",
    "side": "SELL",
    "workingTime": 1670220800315,
    "fills": [],
    "preventedMatches": [
        {
            "preventedMatchId": 0,
            "makerOrderId": 0,
            "price": "1.000000",
            "takerPreventedQuantity": "1.000000"
        }
    ],
    "selfTradePreventionMode": "EXPIRE_TAKER",
    "preventedQuantity": "1.000000"
}

Scénario F – Un utilisateur envoie un ordre au marché EXPIRE_MAKERqui correspondrait à un ordre existant.

Maker Order: symbol=BTCUSDT side=BUY  type=LIMIT  quantity=1 price=1  selfTradePreventionMode=NONE
Taker Order: symbol=BTCUSDT side=SELL type=MARKET quantity=1          selfTradePreventionMode=EXPIRE_MAKER


Résultat : L'ordre existant expire avec le statut EXPIRED_IN_MATCH, en raison du STP. Le nouvel ordre expire également, mais avec le statut EXPIRED, en raison d'une faible liquidité du carnet d'ordres.

Commande du fabricant

{
  "symbol": "BTCUSDT",
  "orderId": 2,
  "orderListId": -1,
  "clientOrderId": "7sgrQQInL69XDMQpiqMaG2",
  "price": "1.000000",
  "origQty": "1.000000",
  "executedQty": "0.000000",
  "cummulativeQuoteQty": "0.000000",
  "status": "EXPIRED_IN_MATCH",
  "timeInForce": "GTC",
  "type": "LIMIT",
  "side": "BUY",
  "stopPrice": "0.000000",
  "icebergQty": "0.000000",
  "time": 1670222557456,
  "updateTime": 1670222557478,
  "isWorking": true,
  "workingTime": 1670222557456,
  "origQuoteOrderQty": "0.000000",
  "selfTradePreventionMode": "NONE",
  "preventedMatchId": 0,
  "preventedQuantity": "1.000000"
}

Ordre du preneur

{
  "symbol": "BTCUSDT",
  "orderId": 3,
  "orderListId": -1,
  "clientOrderId": "zqhsgGDEcdhxy2oza2Ljxd",
  "transactTime": 1670222557478,
  "price": "0.000000",
  "origQty": "1.000000",
  "executedQty": "0.000000",
  "cummulativeQuoteQty": "0.000000",
  "status": "EXPIRED",
  "timeInForce": "GTC",
  "type": "MARKET",
  "side": "SELL",
  "workingTime": 1670222557478,
  "fills": [],
  "preventedMatches": [
    {
      "preventedMatchId": 0,
      "makerOrderId": 2,
      "price": "1.000000",
      "makerPreventedQuantity": "1.000000"
    }
  ],
  "selfTradePreventionMode": "EXPIRE_MAKER"
}

Scénario G - Un utilisateur envoie un ordre limité DECREMENTqui correspondrait à un ordre existant.

Maker Order: symbol=BTCUSDT side=BUY  type=LIMIT quantity=6 price=2  selfTradePreventionMode=NONE
Taker Order: symbol=BTCUSDT side=SELL type=LIMIT quantity=2 price=2  selfTradePreventionMode=DECREMENT


Résultat : les deux commandes ont une quantité empêchée de 2. Étant donné qu'il s'agit de la quantité totale de la commande du preneur, elle expire en raison du STP.

Commande du fabricant

{
  "symbol": "BTCUSDT",
  "orderId": 23,
  "orderListId": -1,
  "clientOrderId": "Kxb4RpsBhfQrkK2r2YO2Z9",
  "price": "2.00000000",
  "origQty": "6.00000000",
  "executedQty": "0.00000000",
  "cummulativeQuoteQty": "0.00000000",
  "status": "NEW",
  "timeInForce": "GTC",
  "type": "LIMIT",
  "side": "BUY",
  "stopPrice": "0.00000000",
  "icebergQty": "0.00000000",
  "time": 1741682807892,
  "updateTime": 1741682816376,
  "isWorking": true,
  "workingTime": 1741682807892,
  "origQuoteOrderQty": "0.00000000",
  "selfTradePreventionMode": "DECREMENT",
  "preventedMatchId": 4,
  "preventedQuantity": "2.00000000"
}

Ordre du preneur

{
  "symbol": "BTCUSDT",
  "orderId": 24,
  "orderListId": -1,
  "clientOrderId": "dwf3qOzD7GM9ysDn9XG9AS",
  "price": "2.00000000",
  "origQty": "2.00000000",
  "executedQty": "0.00000000",
  "cummulativeQuoteQty": "0.00000000",
  "status": "EXPIRED_IN_MATCH",
  "timeInForce": "GTC",
  "type": "LIMIT",
  "side": "SELL",
  "stopPrice": "0.00000000",
  "icebergQty": "0.00000000",
  "time": 1741682816376,
  "updateTime": 1741682816376,
  "isWorking": true,
  "workingTime": 1741682816376,
  "origQuoteOrderQty": "0.00000000",
  "selfTradePreventionMode": "DECREMENT",
  "preventedMatchId": 4,
  "preventedQuantity": "2.00000000"
}
URL réservées aux données de marché
Ces URL ne nécessitent aucune authentification (c'est-à-dire que la clé API n'est pas nécessaire) et ne servent que des données de marché publiques.

 RESTful
Sur l'API RESTful, voici les points de terminaison sur lesquels vous pouvez effectuer des requêtesdata-api.binance.vision :

OBTENIR /api/v3/aggTrades
OBTENIR /api/v3/avgPrice
OBTENIR /api/v3/depth
OBTENIR /api/v3/exchangeInfo
OBTENIR /api/v3/klines
OBTENIR /api/v3/ping
OBTENIR /api/v3/ticker
OBTENIR /api/v3/ticker/24h
OBTENIR /api/v3/ticker/bookTicker
OBTENIR /api/v3/ticker/price
OBTENIR /api/v3/time
OBTENIR /api/v3/trades
OBTENIR /api/v3/uiKlines
Demande d'échantillon :

curl -sX GET "https://data-api.binance.vision/api/v3/exchangeInfo?symbol=BTCUSDT" 


 Websocket
Les données du marché public peuvent également être récupérées via les données du marché WebSocket via l'URL data-stream.binance.vision. Les flux disponibles via ce domaine sont les mêmes que ceux disponibles dans la documentation relative aux flux du marché WebSocket .

Notez que les flux de données utilisateur ne sont pas accessibles via cette URL.

Demande d'échantillon :

wss://data-stream.binance.vision:443/ws/btcusdt@kline_1m

Routage intelligent des commandes (SOR)
Clause de non-responsabilité:

Les symboles et valeurs utilisés ici sont fictifs et n'impliquent rien sur la configuration réelle de l'échange en direct.
Par souci de simplicité, les exemples de ce document n'incluent pas de commission.
Qu'est-ce que le routage intelligent des commandes (SOR)
Le routage intelligent des ordres (SOR) vous permet d'optimiser potentiellement votre liquidité en exécutant un ordre avec des liquidités provenant d'autres carnets d'ordres, avec le même actif de base et des actifs de cotation interchangeables. Les actifs de cotation interchangeables sont des actifs de cotation avec un taux de change fixe de 1 pour 1, comme les stablecoins indexés sur la même monnaie fiduciaire.

Notez que même si les actifs de cotation sont interchangeables, lors de la vente de l'actif de base, vous recevrez toujours l'actif de cotation du symbole dans votre ordre.

Lorsque vous passez une commande à l'aide de SOR, il parcourt les carnets de commandes éligibles, recherche les meilleurs niveaux de prix pour chaque carnet de commandes dans cette configuration SOR et extrait de ces carnets si possible.

Remarque : si la commande utilisant SOR ne peut pas être entièrement exécutée en fonction de la liquidité des carnets de commandes éligibles, LIMIT IOCou MARKETsi les commandes expirent immédiatement, LIMIT GTCla quantité restante sera placée sur le carnet de commandes auquel vous avez initialement soumis la commande.

Exemple 1

Considérons une configuration SOR contenant les symboles BTCUSDT, BTCUSDCet BTCUSDP, et les carnets d'ordres suivants ASK( SELLlatéral) pour ces symboles :

BTCUSDT quantity 3 price 30,800
BTCUSDT quantity 3 price 30,500

BTCUSDC quantity 1 price 30,000
BTCUSDC quantity 1 price 28,000

BTCUSDP quantity 1 price 35,000
BTCUSDP quantity 1 price 29,000

Si vous passez un LIMIT GTC BUYordre pour BTCUSDTavec quantity=0.5et price=31000, vous obtiendrez le meilleur prix de VENTE sur le carnet BTCUSDT à 30 500. Vous dépenserez 15 250 USDT et recevrez 0,5 BTC.

Si vous passez un LIMIT GTC BUYordre en utilisant le SOR pour BTCUSDTavec quantity=0.5et price=31000, vous obtiendrez le meilleur prix de VENTE parmi tous les symboles du SOR , soit BTCUSDC au prix de 28 000. Vous dépenserez 14 000 USDT ( et non USDC !) et recevrez 0,5 BTC.

{
  "symbol": "BTCUSDT",
  "orderId": 2,
  "orderListId": -1,
  "clientOrderId": "sBI1KM6nNtOfj5tccZSKly",
  "transactTime": 1689149087774,
  "price": "31000.00000000",
  "origQty": "0.50000000",
  "executedQty": "0.50000000",
  "cummulativeQuoteQty": "14000.00000000",
  "status": "FILLED",
  "timeInForce": "GTC",
  "type": "LIMIT",
  "side": "BUY",
  "workingTime": 1689149087774,
  "fills": [
    {
      "matchType": "ONE_PARTY_TRADE_REPORT",
      "price": "28000.00000000",
      "qty": "0.50000000",
      "commission": "0.00000000",
      "commissionAsset": "BTC",
      "tradeId": -1,
      "allocId": 0
    }
  ],
  "workingFloor": "SOR",
  "selfTradePreventionMode": "NONE",
  "usedSor": true
}

Exemple 2

En utilisant le même carnet d’ordres que dans l’exemple 1 :

BTCUSDT quantity 3 price 30,800
BTCUSDT quantity 3 price 30,500

BTCUSDC quantity 1 price 30,000
BTCUSDC quantity 1 price 28,000

BTCUSDP quantity 1 price 35,000
BTCUSDP quantity 1 price 29,000

Si vous envoyez une LIMIT GTC BUYcommande pour BTCUSDTavec quantity=5et price=31000, vous devez :

correspondre avec les 3 BTCUSDT à 30 500, et acheter 3 BTC pour 91 500 USDT
puis faites correspondre les 3 BTCUSDT à 30 800 et achetez 2 BTC pour 61 600 USDT
Au total, vous dépensez 153 100 USDT et recevez 5 BTC.

Si vous envoyez la même LIMIT GTC BUYcommande en utilisant SOR pour BTCUSDTavec quantity=5et price=31000, vous devez :

correspondre avec 1 BTCUSDC à 28 000, et acheter 1 BTC pour 28 000 USDT
correspondre avec 1 BTCUSDP à 29 000, et acheter 1 BTC pour 29 000 USDT
match avec 1 BTCUSDC à 30 000, et achetez 1 BTC pour 30 000 USDT
match avec 3 BTCUSDT à 30 500 et achetez 2 BTC pour 61 000 USDT
Au total, vous dépensez 148 000 USDT et recevez 5 BTC.

{
  "symbol": "BTCUSDT",
  "orderId": 2,
  "orderListId": -1,
  "clientOrderId": "tHonoNjWfOSaKiTygN3bfY",
  "transactTime": 1689146154686,
  "price": "31000.00000000",
  "origQty": "5.00000000",
  "executedQty": "5.00000000",
  "cummulativeQuoteQty": "148000.00000000",
  "status": "FILLED",
  "timeInForce": "GTC",
  "type": "LIMIT",
  "side": "BUY",
  "workingTime": 1689146154686,
  "fills": [
    {
      "matchType": "ONE_PARTY_TRADE_REPORT",
      "price": "28000.00000000",
      "qty": "1.00000000",
      "commission": "0.00000000",
      "commissionAsset": "BTC",
      "tradeId": -1,
      "allocId": 0
    },
    {
      "matchType": "ONE_PARTY_TRADE_REPORT",
      "price": "29000.00000000",
      "qty": "1.00000000",
      "commission": "0.00000000",
      "commissionAsset": "BTC",
      "tradeId": -1,
      "allocId": 1
    },
    {
      "matchType": "ONE_PARTY_TRADE_REPORT",
      "price": "30000.00000000",
      "qty": "1.00000000",
      "commission": "0.00000000",
      "commissionAsset": "BTC",
      "tradeId": -1,
      "allocId": 2
    },
    {
      "matchType": "ONE_PARTY_TRADE_REPORT",
      "price": "30500.00000000",
      "qty": "2.00000000",
      "commission": "0.00000000",
      "commissionAsset": "BTC",
      "tradeId": -1,
      "allocId": 3
    }
  ],
  "workingFloor": "SOR",
  "selfTradePreventionMode": "NONE",
  "usedSor": true
}

Exemple 3

En utilisant le même carnet d’ordres que les exemples 1 et 2 :

BTCUSDT quantity 3 price 30,800
BTCUSDT quantity 3 price 30,500

BTCUSDC quantity 1 price 30,000
BTCUSDC quantity 1 price 28,000

BTCUSDP quantity 1 price 35,000
BTCUSDP quantity 1 price 29,000

Si vous envoyez un MARKET BUYordre BTCUSDT utilisant SOR avec quantity=11, seuls 10 BTC sont disponibles au total sur l'ensemble des carnets d'ordres éligibles. Une fois tous les carnets d'ordres configurés en SOR épuisés, la quantité restante (1) expire.

{
  "symbol": "BTCUSDT",
  "orderId": 2,
  "orderListId": -1,
  "clientOrderId": "jdFYWTNyzplbNvVJEzQa0o",
  "transactTime": 1689149513461,
  "price": "0.00000000",
  "origQty": "11.00000000",
  "executedQty": "10.00000000",
  "cummulativeQuoteQty": "305900.00000000",
  "status": "EXPIRED",
  "timeInForce": "GTC",
  "type": "MARKET",
  "side": "BUY",
  "workingTime": 1689149513461,
  "fills": [
    {
      "matchType": "ONE_PARTY_TRADE_REPORT",
      "price": "28000.00000000",
      "qty": "1.00000000",
      "commission": "0.00000000",
      "commissionAsset": "BTC",
      "tradeId": -1,
      "allocId": 0
    },
    {
      "matchType": "ONE_PARTY_TRADE_REPORT",
      "price": "29000.00000000",
      "qty": "1.00000000",
      "commission": "0.00000000",
      "commissionAsset": "BTC",
      "tradeId": -1,
      "allocId": 1
    },
    {
      "matchType": "ONE_PARTY_TRADE_REPORT",
      "price": "30000.00000000",
      "qty": "1.00000000",
      "commission": "0.00000000",
      "commissionAsset": "BTC",
      "tradeId": -1,
      "allocId": 2
    },
    {
      "matchType": "ONE_PARTY_TRADE_REPORT",
      "price": "30500.00000000",
      "qty": "3.00000000",
      "commission": "0.00000000",
      "commissionAsset": "BTC",
      "tradeId": -1,
      "allocId": 3
    },
    {
      "matchType": "ONE_PARTY_TRADE_REPORT",
      "price": "30800.00000000",
      "qty": "3.00000000",
      "commission": "0.00000000",
      "commissionAsset": "BTC",
      "tradeId": -1,
      "allocId": 4
    },
    {
      "matchType": "ONE_PARTY_TRADE_REPORT",
      "price": "35000.00000000",
      "qty": "1.00000000",
      "commission": "0.00000000",
      "commissionAsset": "BTC",
      "tradeId": -1,
      "allocId": 5
    }
  ],
  "workingFloor": "SOR",
  "selfTradePreventionMode": "NONE",
  "usedSor": true
}

Exemple 4

Considérons une configuration SOR contenant les symboles BTCUSDT, BTCUSDCet BTCUSDPet le carnet d'ordres suivant BID( BUYlatéral) pour ces symboles :

BTCUSDT quantity 5 price 29,500

BTCUSDC quantity 5 price 35,000
BTCUSDC quantity 5 price 30,000

BTCUSDP quantity 5 price 28,000

Si vous passez un LIMIT GTC SELLordre pour BTCUSDTavec price=29000et quantity=10, vous vendriez 5 BTC et recevriez 147 500 USDT. Comme il n'existe pas de meilleur prix disponible sur le carnet BTCUSDT, le reste de l'ordre (non exécuté) restera à ce prix de 29 000.

Si vous envoyez une LIMIT GTC SELLcommande en utilisant SOR pour BTCUSDT, vous devez :

match avec 5 BTCUSDC à 35 000 et vendez 5 BTC pour 175 000 USDT
match avec 5 BTCUSDC à 30 000 et vendez 5 BTC pour 150 000 USDT
Au total, vous vendez 10 BTC et recevez 325 000 USDT.

{
  "symbol": "BTCUSDT",
  "orderId": 1,
  "orderListId": -1,
  "clientOrderId": "W1iXSng1fS77dvanQJDGA5",
  "transactTime": 1689147920113,
  "price": "29000.00000000",
  "origQty": "10.00000000",
  "executedQty": "10.00000000",
  "cummulativeQuoteQty": "325000.00000000",
  "status": "FILLED",
  "timeInForce": "GTC",
  "type": "LIMIT",
  "side": "SELL",
  "workingTime": 1689147920113,
  "fills": [
    {
      "matchType": "ONE_PARTY_TRADE_REPORT",
      "price": "35000.00000000",
      "qty": "5.00000000",
      "commission": "0.00000000",
      "commissionAsset": "USDT",
      "tradeId": -1,
      "allocId": 0
    },
    {
      "matchType": "ONE_PARTY_TRADE_REPORT",
      "price": "30000.00000000",
      "qty": "5.00000000",
      "commission": "0.00000000",
      "commissionAsset": "USDT",
      "tradeId": -1,
      "allocId": 1
    }
  ],
  "workingFloor": "SOR",
  "selfTradePreventionMode": "NONE",
  "usedSor": true
}

Résumé : L'objectif du SOR est d'améliorer potentiellement la liquidité des carnets d'ordres grâce à des actifs de cotation interchangeables. Un meilleur accès à la liquidité permet d'exécuter les ordres plus efficacement et à de meilleurs prix pendant la phase de prise d'ordres.

Quels symboles prennent en charge SOR
Vous pouvez trouver la configuration SOR actuelle dans Exchange Information ( GET /api/v3/exchangeInfopour Rest et exchangeInfosur l'API Websocket).

  "sors": [
    {
      "baseAsset": "BTC",
      "symbols": [
        "BTCUSDT",
        "BTCUSDC",
        "BTCUSDP"
      ]
    }
  ]

Ce sorschamp est facultatif. Il est omis dans les réponses si le SOR n'est pas disponible.

Comment passer une commande en utilisant SOR
Sur l'API Rest, la requête est POST /api/v3/sor/order.

Sur l'API WebSocket, la requête est sor.order.place.

Dans la réponse de l'API, il y a un champ appelé workingFloor. Que signifie ce champ
Il s'agit d'un terme utilisé pour déterminer où la dernière activité de la commande a eu lieu (exécution, expiration ou placement comme nouvelle commande, etc.).

Si le workingFloorest SOR, cela signifie que votre commande a interagi avec d'autres carnets de commandes éligibles dans la configuration SOR.

Si le workingFloorest EXCHANGE, cela signifie que votre commande a interagi avec le carnet de commandes auquel vous avez envoyé cette commande.

Dans la réponse de l'API, fillsles champs matchTypeet allocId. Que signifient-ils
matchTypele champ indique un remplissage de commande non standard.

Lorsque votre ordre est exécuté par SOR, vous verrez matchType: ONE_PARTY_TRADE_REPORT, indiquant que vous n'avez pas négocié directement sur la bourse ( tradeId: -1). Votre ordre est exécuté par allocations .

allocIdle champ identifie l'allocation afin que vous puissiez l'interroger ultérieurement.

Que sont les allocations 
Une allocation est un transfert d'actif de la bourse vers votre compte. Par exemple, lorsque SOR prélève des liquidités dans les carnets d'ordres éligibles, votre ordre est exécuté par allocations. Dans ce cas, vous ne négociez pas directement, mais recevez des allocations de SOR correspondant aux transactions effectuées par SOR pour votre compte.

[
  {
    "symbol": "BTCUSDT",            // Symbol the order was submitted to
    "allocationId": 0,    
    "allocationType": "SOR",
    "orderId": 2,       
    "orderListId": -1,
    "price": "30000.00000000",      // Price of the fill
    "qty": "5.00000000",            // Quantity of the fill
    "quoteQty": "150000.00000000",
    "commission": "0.00000000",
    "commissionAsset": "BTC",
    "time": 1688379272280,          // Time the allocation occurred
    "isBuyer": true,
    "isMaker": false,
    "isAllocator": false
  }
]


Comment interroger les commandes qui ont utilisé SOR
Vous pouvez les trouver de la même manière que pour toute autre commande. La principale différence réside dans le fait que la réponse d'une commande utilisant SOR comporte deux champs supplémentaires : usedSoret workingFloor.

Comment puis-je obtenir des détails sur mes exécutions pour les commandes qui ont utilisé SOR
Lorsque les ordres SOR sont négociés sur des carnets d'ordres autres que le symbole soumis avec l'ordre, celui-ci est exécuté avec une allocation et non une transaction. Les ordres passés avec SOR peuvent potentiellement comporter à la fois des allocations et des transactions.

Dans la réponse de l'API, vous pouvez consulter les fillschamps. Les allocations ont un allocIdet "matchType": "ONE_PARTY_TRADE_REPORT", tandis que les transactions ont un tradeId.

Les allocations peuvent être interrogées à l'aide de GET /api/v3/myAllocations(API Rest) ou myAllocations(API WebSocket).

Les transactions peuvent être interrogées à l'aide de GET /api/v3/myTrades(API Rest) ou myTrades(API WebSocket).


Règles de comptage des commandes non exécutées au comptant
Afin de garantir un marché spot équitable et ordonné, nous limitons le rythme auquel de nouvelles commandes peuvent être passées.

La limite de taux s'applique au nombre de nouvelles commandes non exécutées passées au cours d'un intervalle de temps donné. Autrement dit, les commandes partiellement ou totalement exécutées ne sont pas comptabilisées dans la limite de taux.

[!NOTE]
La limite du taux de commandes non exécutées récompense les traders efficaces.

Tant que vos ordres sont négociés, vous pouvez continuer à négocier.

Plus d'informations : Comment les commandes exécutées affectent-elles la limite de taux ?

Quelles sont les limites de taux actuelles
Vous pouvez interroger les limites de taux actuelles à l'aide de la demande « informations sur l'échange ».

Indique "rateLimitType": "ORDERS"la limite actuelle du taux de commandes non exécutées.

Veuillez vous référer à la documentation de l'API :

API	Demande
API FIX	LimitQuery<XLQ>
API REST	GET /api/v3/exchangeInfo
API WebSocket	exchangeInfo
[!IMPORTANT]
Les demandes de placement de commande sont également affectées par les limites générales de taux de requêtes sur les API REST et WebSocket et les limites de messages sur l'API FIX.

Si vous envoyez trop de requêtes à un rythme élevé, vous serez bloqué par l'API.


Comment fonctionne la ORDERSlimite de taux non remplie
Chaque demande de commande réussie s'ajoute au nombre de commandes non exécutées pour l'intervalle de temps en cours. Si trop de commandes non exécutées s'accumulent pendant cet intervalle, les demandes suivantes seront rejetées.

Par exemple, si la limite du taux de commandes non exécutées est de 100 toutes les 10 secondes :

{
  "rateLimitType": "ORDERS",
  "interval": "SECOND",
  "intervalNum": 10,
  "limit": 100
}

vous pouvez alors passer au maximum 100 nouvelles commandes entre 12:34:00 et 12:34:10, puis 100 autres de 12:34:10 à 12:34:20, et ainsi de suite.

[!TIP]
Si les commandes nouvellement passées sont exécutées, votre nombre de commandes non exécutées diminue et vous pouvez passer davantage de commandes pendant l'intervalle de temps.

Plus d'informations : Comment les commandes exécutées affectent-elles la limite de taux ?

Lorsqu'une commande est rejetée par le système en raison de la limite du taux de commandes non exécutées, le code d'état HTTP est défini sur 429 Too Many Requestset le code d'erreur est -1015 "Too many new orders".

Si vous rencontrez ces erreurs, veuillez arrêter d'envoyer des commandes jusqu'à l'expiration de l'intervalle de limite de débit affecté.

Veuillez vous référer à la documentation de l'API :

API	Documentation
API FIX	Nombre de commandes non exécutées
API REST	Nombre de commandes non exécutées
API WebSocket	Nombre de commandes non exécutées
Le nombre de commandes non exécutées est-il suivi par adresse IP
Le nombre de commandes non exécutées est suivi par (sous-)compte .

Le nombre de commandes non exécutées est partagé entre toutes les adresses IP, toutes les clés API et toutes les API.


Comment les commandes exécutées affectent-elles le nombre de commandes non exécutées
Lorsqu'un ordre est exécuté pour la première fois (partiellement ou intégralement), le nombre d'ordres non exécutés est décrémenté d'un ordre pour chaque intervalle de la ORDERSlimite de taux. En effet, les ordres négociés ne sont pas comptabilisés dans la limite de taux, ce qui permet aux traders efficaces de passer de nouveaux ordres.

Certaines commandes offrent des incitations supplémentaires :

Commandes qui ne sont pas remplies immédiatement (c'est-à-dire remplies en premier lors de la phase de création).
Commandes qui remplissent de grandes quantités.
Dans ces cas, le nombre d'ordres non exécutés peut être diminué de plus d'un ordre pour chaque ordre qui commence à être négocié.

Remarques :

Les exemples ne donnent qu'une idée générale du comportement. L'intervalle de 10 secondes est utilisé pour plus de simplicité. La configuration réelle sur l'échange en direct peut être différente.
Il y a un léger délai entre l'exécution de la commande et la mise à jour du nombre de commandes non exécutées. Soyez vigilant lorsque le nombre de commandes non exécutées approche de la limite.
Veuillez vous référer à Comment ORDERSfonctionne la limite de taux de commandes non exécutées ? pour voir comment vous pouvez surveiller le nombre de commandes non exécutées en fonction de l'API.
Exemple 1 — preneur :

Temps	Action	Nombre de commandes non exécutées
00:00:00		0
00:00:01	Passer une commande LIMITE A	1 — nouvelle commande (+1)
00:00:02	Passer une commande LIMITE B	2 — nouvelle commande (+1)
(commande B partiellement remplie)	1 — premier remplissage en tant que preneur (−1)
00:00:03	Passer une commande LIMITE C	2 — nouvelle commande (+1)
00:00:04	(commande B partiellement remplie)	2
00:00:04	(commande B exécutée)	2
00:00:05	Passer une commande MARKET D	3 — nouvelle commande (+1)
(commande D entièrement remplie)	2 — premier remplissage en tant que preneur (−1)
Notez que pour chaque ordre de preneur qui est immédiatement négocié, le nombre d'ordres non exécutés est décrémenté ultérieurement, vous permettant de continuer à passer des ordres.

Exemple 2 — fabricant :

Temps	Action	Nombre de commandes non exécutées
00:00:00		0
00:00:01	Passer une commande LIMITE A	1 — nouvelle commande (+1)
00:00:01	Passer une commande LIMITE B	2 — nouvelle commande (+1)
00:00:02	Passer une commande LIMITE C	3 — nouvelle commande (+1)
00:00:02	Passer une commande LIMITE D	4 — nouvelle commande (+1)
00:00:02	Passer une commande LIMITE E	5 — nouvelle commande (+1)
00:00:03	(commande A partiellement remplie)	0 — premier remplissage en tant que créateur (−5)
00:00:04	Passer une commande LIMITE F	1 — nouvelle commande (+1)
00:00:04	Passer une commande LIMIT G	2 — nouvelle commande (+1)
00:00:05	(commande A partiellement remplie)	2
00:00:05	(commande A remplie)	2
00:00:05	(commande B partiellement remplie)	0 — premier remplissage en tant que créateur (−5)
00:00:06	Passer une commande LIMIT H	1 — nouvelle commande (+1)
Notez que pour chaque commande de fabricant exécutée ultérieurement, le nombre de commandes non exécutées est diminué d'un montant plus élevé, vous permettant de passer davantage de commandes.

Comment les commandes annulées ou expirées affectent-elles le nombre de commandes non exécutées
L'annulation d'une commande ne modifie pas le nombre de commandes non exécutées.

Les commandes expirées ne modifient pas non plus le nombre de commandes non exécutées.

Exemple:

Temps	Action	Nombre de commandes non exécutées
00:00:00		0
00:00:01	Passer une commande LIMITE A	1 — nouvelle commande (+1)
00:00:02	Annuler la commande A	1
00:00:02	Passer une commande LIMITE B	2 — nouvelle commande (+1)
00:00:03	Passer une commande LIMIT FOK C	3 — nouvelle commande (+1)
(la commande C est entièrement remplie)	2 — remplir (−1)
00:00:05	Passer une commande LIMITE D	3 — nouvelle commande (+1)
00:00:06	Passer une commande LIMIT FOK E	4 — nouvelle commande (+1)
(la commande E expire sans être remplie)	4
00:00:07	Annuler la commande D	4
00:00:07	Passer une commande LIMITE F	5 — nouvelle commande (+1)
Quel fuseau horaire "interval":"DAY"utilisez-vous
UTC

Que se passe-t-il si j'ai passé une commande hier mais qu'elle est exécutée le lendemain
Les nouvelles commandes exécutées diminuent votre nombre actuel de commandes non exécutées, quelle que soit la date à laquelle les commandes ont été passées.

Exemple:

Temps	Action	Nombre de commandes non exécutées
01/01/2024 09:00	Passer 5 commandes : 1..5	5
02/01/2024 00:00	(réinitialisation de l'intervalle de limite de débit)	0
02/01/2024 09:00	Passer 10 commandes : 6..15	10
02/01/2024 12:00	(les commandes 1..5 sont remplies)	5
02/01/2024 13:00	(les commandes 6..10 sont remplies)	0
02/01/2024 14:00	Passer 2 commandes : 16, 17	2
02/01/2024 15:00	(les commandes 11..15 sont remplies)	0
Remarque : Vous ne recevez aucun crédit pour les commandes exécutées. Autrement dit, une fois le nombre de commandes non exécutées à zéro, les exécutions supplémentaires ne le réduiront pas davantage. Les nouvelles commandes augmenteront le nombre de commandes comme d'habitude.



FAQ sur le codage binaire simple (SBE)
L'objectif de ce document est d'expliquer :

Comment recevoir les réponses SBE dans l'API SPOT.
Comment décoder les réponses SBE.
SBE est un format de sérialisation utilisé pour une faible latence.

Cette implémentation est basée sur la spécification FIX SBE.

Dépôt GitHub
document HTML
Comment obtenir une 
 REST
L' Accepten-tête doit inclure application/sbe.
Fournissez l'ID de schéma et la version dans l' X-MBX-SBEen-tête sous la forme <ID>:<VERSION>.
Exemple de demande (REST) :

curl -sX GET -H "Accept: application/sbe" -H "X-MBX-SBE: 1:0" 'https://api.binance.com/api/v3/exchangeInfo?symbol=BTCUSDT'


Remarques :

Si vous fournissez uniquement application/sbedans l'en-tête Accepter :
Si SBE n'est pas activé dans l'échange, vous recevrez un HTTP 406 Not Acceptable .
Si les informations <ID>:<VERSION>fournies dans l' X-MBX-SBEen-tête sont mal formées ou non valides, la réponse sera une erreur codée SBE.
Si l' X-MBX-SBEen-tête est manquant, la réponse sera une erreur codée SBE.
Si vous fournissez les deux application/sbedans application/jsonl'en-tête Accepter :
Si SBE n’est pas activé dans l’échange, la réponse reviendra à JSON.
Si les informations <ID>:<VERSION>fournies dans l' X-MBX-SBEen-tête sont mal formées ou non valides, la réponse reviendra au format JSON.
Si l' X-MBX-SBEen-tête est manquant, la réponse reviendra à JSON.
 WebSocket
Dans l'URL de connexion, ajoutez responseFormat=sbe.
Fournissez l'ID de schéma et la version dans les paramètres sbeSchemaId=<SCHEMA_ID>et sbeSchemaVersion=<SCHEMA_VERSION>respectivement.
Exemple de demande (WebSocket) :

id=$(date +%s%3N)
method="exchangeInfo"
params='{"symbol":"BTCUSDT"}'

request=$( jq -n \
        --arg id "$id" \
        --arg method "$method" \
        --argjson params "$params" \
        '{id: $id, method: $method, params: $params}' )

response=$(echo $request | websocat -n1 'wss://ws-api.binance.com:443/ws-api/v3?responseFormat=sbe&sbeSchemaId=1&sbeSchemaVersion=0')


Remarques :

Si vous fournissez uniquement responseFormat=sbedans l'URL de connexion :
Si SBE n'est pas activé dans l'échange, la réponse sera HTTP 400.
Si les sbeSchemaId=<SCHEMA_ID>ou sbeSchemaVersion=<SCHEMA_VERSION>sont mal formés ou invalides, la réponse sera HTTP 400.
Si vous fournissez à la fois responseFormat=sbeet responseFormat=json, la réponse sera HTTP 400.
Toutes les réponses d'erreur pendant la négociation HTTP sont codées au format JSON avec l' Content-Typeen-tête défini sur application/json;charset=UTF-8.
Une fois qu'une session WebSocket a été établie avec succès avec SBE activé, toutes les réponses de méthode au sein de cette session sont codées en SBE, même dans le cas où SBE est désactivé.
Cela signifie que si SBE est désactivé alors que votre connexion WebSocket est active, vous recevrez une erreur « SBE n'est pas activé » codée SBE en réponse à toute demande ultérieure.
Au moment de la rédaction de ce document, nous déconseillons son utilisation websocatpour envoyer des requêtes, car nous avons constaté des problèmes de décodage des trames binaires. L'exemple ci-dessus est uniquement utilisé à titre de référence pour afficher l'URL permettant d'obtenir une réponse SBE.
 prises en charge
L'API REST et l'API WebSocket pour SPOT prennent en charge SBE.

 SBE
Le schéma à utiliser à la fois pour l'échange en direct et pour SPOT Testnet sera enregistré dans ce référentiel ici .
Toutes les mises à jour du schéma seront notées dans le CHANGELOG .
Concernant le support Legacy :

Les schémas SBE sont versionnés via deux attributs XML, idet version.
idest incrémenté lors d'une modification radicale. Dans ce cas, versionil est réinitialisé à 0.
versionest incrémenté lorsqu'une modification non-sécante est introduite. Dans ce cas, idil n'est pas modifié.
Lorsqu'un nouveau schéma est mis en ligne, l'ancien schéma devient obsolète. Cette désapprobation se produit même si le nouveau schéma n'introduit que des modifications non-ruptures.
Un schéma obsolète sera pris en charge pendant au moins six mois après sa dépréciation .
Par exemple, selon ce calendrier hypothétique :
30 janvier 2024 : La version 0 du schéma ID 1 est publiée. Il s'agit de la première version, elle sera donc utilisable une fois SBE activé dans la plateforme Exchange.
30 mars 2024 : publication de la version 1 du schéma ID 1. Ce schéma introduit une modification non-sinistre.
L'ID de schéma 1 version 0 est obsolète, mais peut encore être utilisé pendant au moins 6 mois supplémentaires.
30 août 2024 : publication de la version 0 du schéma ID 2. Ce schéma introduit une modification radicale.
L'ID de schéma 1 version 0 est obsolète, mais peut encore être utilisé pendant au moins 1 mois supplémentaire.
Le schéma id 1 version 1 est obsolète, mais peut encore être utilisé pendant au moins 6 mois supplémentaires.
30 septembre 2024 : 6 mois se sont écoulés depuis la sortie de la version 1 de Schema id 1.
L'ID de schéma 1 version 0 est retiré.
30 février 2025 : publication de la version 1 du schéma ID 2. Ce schéma introduit une modification non-sinistre.
L'ID de schéma 1 version 1 est retiré.
Le schéma id 2 version 0 est obsolète, mais peut encore être utilisé pendant au moins 6 mois supplémentaires.
Les réponses HTTP contiendront un X-MBX-SBE-DEPRECATEDen-tête pour les requêtes spécifiant un obsolète <ID>:<VERSION>dans leur X-MBX-SBEen-tête.
Pour les réponses WebSocket, le champ sbeSchemaIdVersionDeprecatedsera défini sur truepour les requêtes spécifiant une valeur obsolète sbeSchemaIdet sbeSchemaVersiondans leur URL de connexion.
<ID>:<VERSION>Les requêtes spécifiant une API REST ou sbeSchemaIdune sbeSchemaVersion API WebSocket retirée échoueront avec HTTP 400.
Le fichier JSON relatif au cycle de vie des schémas, avec les dates des schémas les plus récents, obsolètes et supprimés pour l'échange en direct et le réseau de test SPOT, sera enregistré dans ce référentiel . Voici
un exemple JSON basé sur la chronologie hypothétique ci-dessus :
{
    "environment": "PROD",
    "latestSchema": {
        "id": 2,
        "version": 1,
        "releaseDate": "3025-02-01" 
    },
    "deprecatedSchemas": [
        {
            "id": 2,
            "version": 0,
            "releaseDate": "3024-08-01",
            "deprecatedDate": "3025-02-01" 
        }
    ],
    "retiredSchemas": [
        {
            "id": 1,
            "version": 1,
            "releaseDate": "3024-03-01",
            "deprecatedDate": "3024-08-01", 
            "retiredDate": "3025-02-01",
        },
        {
            "id": 1,
            "version": 0,
            "releaseDate": "3024-01-01",
            "deprecatedDate": "3024-03-01",
            "retiredDate": "3024-09-01",
        }
    ]
}

Générer des décodeurs SBE
Télécharger le schéma :
spot_prod_latest.xmlpour l'échange en direct.
spot_testnet_latest.xmlpour le réseau de test SPOT .
Cloner et construire simple-binary-encoding:
 $ git clone https://github.com/real-logic/simple-binary-encoding.git
 $ cd simple-binary-encoding
 $ ./gradlew


Exécutez le générateur de code SbeTool. (Voici des exemples de décodage de la charge utile d'Exchange Information en Java , C++ et Rust .)
 des champs décimaux
Contrairement à la spécification FIX SBE, les champs décimaux ont leurs champs de mantisse et d'exposant codés séparément en tant que champs primitifs afin de minimiser la taille de la charge utile et le nombre de champs codés dans les messages.

 du champ d'horodatage
Les horodatages des réponses SBE sont exprimés en microsecondes. Ceci diffère des réponses JSON, qui contiennent des horodatages en millisecondes.

Attributs de champ personnalisés dans le 
Quelques attributs de champ préfixés par mbx:ont été ajoutés au fichier de schéma à des fins de documentation :

mbx:exponent:Pointe vers le champ d'exposant correspondant au champ de mantisse
mbx:jsonPath:Contient le nom du champ équivalent dans la réponse JSON
mbx:jsonValue: Contient le nom de la valeur ENUM équivalente dans la réponse JSON


Types de clés API
Les API Binance nécessitent une clé API pour accéder aux points de terminaison authentifiés pour le trading, l'historique du compte, etc.

Nous prenons en charge plusieurs types de clés API :

Ed25519 (recommandé)
HMAC
RSA
Ce document fournit un aperçu des clés API prises en charge.

Nous vous recommandons d'utiliser les clés API Ed25519 car elles devraient offrir les meilleures performances et la meilleure sécurité parmi tous les types de clés pris en charge.

Lisez la documentation de l’API REST ou de l’API WebSocket pour savoir comment utiliser différentes clés API.

Les clés Ed25519 utilisent la cryptographie asymétrique. Vous partagez votre clé publique avec Binance et utilisez la clé privée pour signer les requêtes API. L'API Binance utilise la clé publique pour vérifier votre signature.

Les clés Ed25519 offrent une sécurité comparable aux clés RSA 3072 bits, mais avec une clé considérablement plus petite, une taille de signature plus petite et un calcul de signature plus rapide.

Nous vous recommandons d'utiliser les clés API Ed25519.

Exemple de clé Ed25519 :

-----BEGIN PUBLIC KEY-----
MCowBQYDK2VwAyEAgmDRTtj2FA+wzJUIlAL9ly1eovjLBu7uXUFR+jFULmg=
-----END PUBLIC KEY-----

Exemple de signature Ed25519 :

E7luAubOlcRxL10iQszvNCff+xJjwJrfajEHj1hOncmsgaSB4NE+A/BbQhCWwit/usNJ32/LeTwDYPoA7Qz4BA==


Les clés HMAC utilisent la cryptographie symétrique. Binance génère et partage avec vous une clé secrète que vous utilisez pour signer les requêtes API. L'API Binance utilise cette même clé secrète partagée pour vérifier votre signature.

Les signatures HMAC sont rapides à calculer et compactes.
Cependant, le secret partagé doit être partagé entre plusieurs parties, ce qui est moins sécurisé que la cryptographie asymétrique utilisée par les clés Ed25519 ou RSA.

Les clés HMAC sont obsolètes. Nous recommandons de migrer vers des clés API asymétriques, telles que Ed25519 ou RSA.

Exemple de clé HMAC :

Fhs4lGae2qAi6VNjbJjebUAwXrIChb7mlf372UOICMwdKaNdNBGKtfdeUff2TTTT

Exemple de signature HMAC :

7f3fc79c57d7a70d2b644ad4589672f4a5d55a62af2a336a0af7d4896f8d48b8

Les clés RSA utilisent la cryptographie asymétrique.
Vous partagez votre clé publique avec Binance et utilisez la clé privée pour signer les requêtes API.
L'API Binance utilise la clé publique pour vérifier votre signature.

Nous prenons en charge les clés RSA 2048 et 4096 bits.

Bien que les clés RSA soient plus sécurisées que les clés HMAC, les signatures RSA sont beaucoup plus grandes que HMAC et Ed25519, ce qui peut entraîner une dégradation des performances.

Exemple de clé RSA (2048 bits) :

-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAyfKiFXpcOhF5rX1XxePN
akwN7Etwtn3v05cZNY+ftDHbVZHs/kY6Ruj5lhxVFAq5dv7Ba9/4jPijXuMuIc6Y
8nUlqtrrxC8DEOAczw9SKATDYZN9nbLfYlbBFfHzRQUXdAtYCPI6XtxmJBS7aOBb
4nZe1SVm+bhLrp0YQnx2P0s+37qkGeVn09m6w9MnWxjgCkkYFPWQkXIu5qOnwx6p
NfqDmFD7d7dUc/6PZQ1bKFALu/UETsobmBk82ShbrBhlc0JXuhf9qBR7QASjHjFQ
2N+VF2PfH8dm5prZIpz/MFKPkBW4Yuss0OXiD+jQt1J2JUKspLqsIqoXjHQQGjL7
3wIDAQAB
-----END PUBLIC KEY-----

Exemple de signature RSA (2048 bits) :

wS6q6h77AvH1TqwInoTDdWIIubRCiUP4RLG++GI24twL3BMtX0EEV+YT1eH8Hb8bLe0Rb9OhOHbt1CC3aurzoCTgZvhNek47mg+Bpu8fwQ7eRkXEiWBx5C8BNN73JwnnkZw4UzYvqiwAs162jToV8AL0eN043KJ3MEKCy3C6nyeYOFSg+1Cp637KtAZk3z7aHknSu7/PXSPuwMIpBgFctf8YKGZFAVRbgwlcgUDhXyaGts6OFePGy0jkZKJHawb/w5hoatatsfVmVC4hZ8fsfystQ9k5DNjTm7ROApWaXy9BsfAYcj13O424mqlpkKG4EGnIjOIWB/pRDDQEm2O/xg==



Commander Modifier Garder la priorité
Clause de non-responsabilité :

Les symboles et valeurs utilisés ici sont fictifs et n'impliquent rien sur la configuration réelle de l'échange en direct.
Par souci de simplicité, les exemples de ce document n'incluent pas de commission.
Qu'est-ce que l'ordre, la modification et le maintien de la priorité
La demande de modification de commande et de maintien de la priorité est utilisée pour modifier (amender) une commande existante sans perdre la priorité du carnet de commandes .

Les modifications de commande suivantes sont autorisées :

réduire la quantité de la commande
Comment puis-je modifier la quantité de ma commande 
Utilisez les requêtes suivantes :

API	Demande
API REST	PUT /api/v3/order/amend/keepPriority
API WebSocket	order.amend.keepPriority
API FIX	Demande de modification de commande et de maintien de la priorité<XAK>
Quelle est la différence entre « Annuler une commande existante et envoyer une nouvelle commande » (annuler-remplacer) et « Modifier la commande et conserver la priorité »
La demande « Annuler un ordre existant et en envoyer un nouveau » annule l'ancien ordre et en place un nouveau.
La priorité temporelle est perdue. Le nouvel ordre est exécuté après les ordres existants au même prix.

La demande de modification d'ordre « Conserver la priorité » modifie un ordre existant.
L'ordre modifié conserve sa priorité temporelle parmi les ordres existants au même prix.

Prenons par exemple le carnet de commandes suivant :

Utilisateur	ID de commande	Côté	Prix de la commande	quantité
Utilisateur A	10	ACHETER	87 000	1,00
⭐️ TOI	15	ACHETER	87 000	5,50
Utilisateur B	20	ACHETER	87 000	4,00
Utilisateur C	21	ACHETER	86 999	2,00
Votre commande 15 est la deuxième dans la file d'attente en fonction du prix et du délai.

Vous souhaitez réduire la quantité de 5,50 à 5,00.

Si vous utilisez annuler-remplacer pour annuler orderId=15et passer une nouvelle commande avec qty=5.00, le carnet de commandes ressemblera à ceci :

Utilisateur	ID de commande	Côté	Prix de la commande	quantité
Utilisateur A	10	ACHETER	87 000	1,00
⭐️ TOI	11	ACHETER	87 000	5,50
Utilisateur B	20	ACHETER	87 000	4,00
⭐️ TOI	(nouveau) 22	ACHETER	87 000	5,00
Utilisateur C	21	ACHETER	86 999	2,00
Notez que la nouvelle commande reçoit un nouvel ID de commande et que vous perdez la priorité temporelle : la commande 22 sera négociée après la commande 20.

Si, au lieu de cela, vous utilisez Order Amend Keep Priority pour réduire la quantité de orderId=15jusqu'à qty=5.00, le carnet de commandes ressemblera à ceci :

Utilisateur	ID de commande	Côté	Prix de la commande	quantité
Utilisateur A	10	ACHETER	87 000	1,00
⭐️ TOI	15	ACHETER	87 000	(modifié) 5,00
Utilisateur B	20	ACHETER	87 000	4,00
Utilisateur C	21	ACHETER	86 999	2,00
Notez que l'identifiant de commande reste le même et que la commande conserve sa priorité dans la file d'attente. Seule la quantité change.

La modification de la commande et le maintien de la priorité ont-ils une incidence sur le nombre de commandes non exécutées (limites de taux)
Actuellement, les demandes de modification de commande et de maintien de la priorité facturent 0 pour le nombre de commandes non exécutées.

Comment savoir si ma commande a été modifiée 
Si la commande a été modifiée avec succès, la réponse de l'API contient votre commande avec la quantité mise à jour.

Sur User Data Stream, vous recevrez un "executionReport"événement avec le type d'exécution "x": "REPLACED".

Si la commande modifiée appartient à une liste de commandes et que l'ID de commande client a changé, vous recevrez également un événement « listStatus » avec le type de statut de liste "l": "UPDATED".

Vous pouvez également utiliser les requêtes suivantes pour interroger l’historique des modifications de commande :

API	Demande
API REST	GET /api/v3/order/amendments
API WebSocket	order.amendments
Que se passe-t-il si ma demande de modification échoue
Si la demande échoue pour une raison quelconque (par exemple, les filtres, les autorisations, les restrictions de compte, etc.) échouent, la demande de modification de commande est rejetée et la commande reste inchangée.

Est-il possible de réutiliser le clientOrderId actuel pour ma commande modifiée
Oui.

Par défaut, les commandes modifiées reçoivent un nouvel ID de commande client aléatoire, mais vous pouvez transmettre l'ID de commande client actuel dans le newClientOrderIdparamètre si vous souhaitez le conserver.

Les ordonnances Iceberg peuvent-elles être modifiées
Oui.

Notez que la quantité visible d'une commande iceberg ne changera que si newQtyelle est inférieure à la quantité visible pré-modifiée.

Les listes de commandes peuvent-elles être modifiées
Les commandes dans une liste de commandes peuvent être modifiées.

Notez que les paires d'ordres OCO doivent avoir la même quantité, car un seul ordre peut être exécuté. Par conséquent, la modification de l'un des ordres affecte les deux ordres.

Pour les commandes OTO, les commandes en cours et en attente peuvent être modifiées individuellement.

Quels symboles permettent de commander, de modifier et de conserver la priorité
Ces informations sont disponibles dans la section Informations sur la Bourse. Les symboles autorisant les demandes de modification d'ordre et de maintien de la priorité sont amendAlloweddéfinis sur true.


