import { Injectable } from '@angular/core';
import { HttpClient, HttpErrorResponse, HttpHeaders } from '@angular/common/http';
import { Observable, throwError, of } from 'rxjs';
import { catchError, map, tap, switchMap, timeout } from 'rxjs/operators';
import { MatSnackBar } from '@angular/material/snack-bar';
import { environment } from '../../environments/environment';
import { HealthService } from './health.service';
import { 
  CardValidationRequest, 
  CardValidationResponse,
  CardGenerationRequest,
  CardGenerationResponse,
  ApiErrorResponse
} from '../models/api.model';
import { Card } from '../models/card.model';

/**
 * Service for card generation and management
 */
@Injectable({
  providedIn: 'root'
})
export class CardService {
  /**
   * Default HTTP headers for Lambda proxy
   */
  private readonly defaultHeaders = new HttpHeaders({
    'Content-Type': 'application/json',
    'ngrok-skip-browser-warning': 'true'
  });

  constructor(
    private http: HttpClient,
    private healthService: HealthService,
    private snackBar: MatSnackBar
  ) { }

  // Note: Automatic retries disabled to prevent duplicate card generations

  /**
   * Get the appropriate timeout value based on whether this is a cold start or warm run
   */
  private getDynamicTimeout(): number {
    const isFirstJobCompleted = this.healthService.isFirstJobCompleted();
    
    if (isFirstJobCompleted) {
      // Warm run - use shorter timeout
      console.log(`🚀 Using warm run timeout: ${environment.warmRunTimeoutMs}ms`);
      return environment.warmRunTimeoutMs;
    } else {
      // Cold start - use longer timeout
      console.log(`🔥 Using cold start timeout: ${environment.coldStartTimeoutMs}ms`);
      return environment.coldStartTimeoutMs;
    }
  }

  /**
   * Validate a card against Magic: The Gathering rules
   * @param card Card object to validate
   * @returns Observable with validation results
   */
  validateCard(card: Card): Observable<CardValidationResponse> {
    // For now, return a mock validation response since we don't have a validation endpoint
    // This can be implemented later when the validation service is available
    const mockResponse: CardValidationResponse = {
      isValid: true,
      errors: [],
      warnings: []
    };
    
    return of(mockResponse);
  }

  /**
   * Generate a complete card using single unified endpoint
   * @param card Card data for generation
   * @returns Observable with the complete card
   */
  generateCompleteCard(card: Card): Observable<Card> {
    const url = `${environment.apiUrl}${environment.cardGenerationEndpoint}`;
    
    const request: CardGenerationRequest = {
      prompt: card.artPrompt || `Fantasy art of ${card.name}, ${card.type}`,
      width: environment.defaultCardWidth,
      height: environment.defaultCardHeight,
      cardData: {
        name: card.name,
        manaCost: card.manaCost,
        supertype: card.supertype,
        colors: card.colors,
        type: card.type,
        subtype: card.subtype,
        rarity: card.rarity,
        cmc: card.cmc,
        description: card.description,
        power: card.power,
        toughness: card.toughness
      }
    };

    console.log('Frontend sending colors to backend:', card.colors);
    console.log('Full cardData being sent:', request.cardData);

    return this.http.post<CardGenerationResponse>(url, request, { headers: this.defaultHeaders })
      .pipe(
        timeout(this.getDynamicTimeout()),
        map((response: any) => {
          const updatedCard: Card = { ...card };
          
          // Process card data if available
          if (response.cardData) {
            try {
              // Try to parse structured card data or use as description
              const parsedData = JSON.parse(response.cardData);
              if (parsedData.name) updatedCard.name = parsedData.name;
              if (parsedData.description) updatedCard.description = parsedData.description;
              if (parsedData.flavorText) updatedCard.flavorText = parsedData.flavorText;
              if (parsedData.manaCost) updatedCard.manaCost = parsedData.manaCost;
            } catch (e) {
              // If not JSON, treat as description
              updatedCard.description = response.cardData;
            }
          }
          
          // Process image data if available
          if (response.imageData) {
            updatedCard.imageUrl = `data:image/png;base64,${response.imageData}`;
          }
          
          // Process complete card image if available
          if (response.card_image) {
            updatedCard.cardImageUrl = `data:image/png;base64,${response.card_image}`;
          }
          
          return updatedCard;
        }),
        tap(card => console.log('Unified card generation successful:', card)),
        catchError(this.handleError<Card>('generateCompleteCard'))
      );
  }

  /**
   * Generate only card text content using fast text-only endpoint
   * @param card Card data for text generation
   * @returns Observable with updated card text
   */
  generateCardTextOnly(card: Card): Observable<Partial<Card>> {
    const url = `${environment.apiUrl}/api/v1/create_card_text_only`;
    
    const request: CardGenerationRequest = {
      prompt: card.artPrompt || `Fantasy art of ${card.name}, ${card.type}`,
      width: environment.defaultCardWidth,
      height: environment.defaultCardHeight,
      cardData: {
        name: card.name,
        manaCost: card.manaCost,
        supertype: card.supertype,
        colors: card.colors,
        type: card.type,
        subtype: card.subtype,
        rarity: card.rarity,
        cmc: card.cmc,
        description: card.description,
        power: card.power,
        toughness: card.toughness
      }
    };

    return this.http.post<any>(url, request, { headers: this.defaultHeaders })
      .pipe(
        timeout(30000), // Use shorter timeout for text-only (30 seconds)
        map((response: any) => {
          const updatedCardData: Partial<Card> = {};
          
          // Process only card data (text-only response)
          if (response.cardData) {
            try {
              // Try to parse structured card data or use as description
              const parsedData = JSON.parse(response.cardData);
              if (parsedData.name) updatedCardData.name = parsedData.name;
              if (parsedData.description) updatedCardData.description = parsedData.description;
              if (parsedData.flavorText) updatedCardData.flavorText = parsedData.flavorText;
              if (parsedData.manaCost) updatedCardData.manaCost = parsedData.manaCost;
            } catch (e) {
              // If not JSON, treat as description
              updatedCardData.description = response.cardData;
            }
          }
          
          return updatedCardData;
        }),
        tap(cardData => console.log('Fast text-only generation successful:', cardData)),
        catchError(this.handleError<Partial<Card>>('generateCardTextOnly'))
      );
  }

  /**
   * Regenerate card text using existing image data
   * @param card Card data for text generation
   * @param imageData Base64 image data to reuse
   * @returns Observable with updated card text
   */
  regenerateCardText(card: Card, imageData: string): Observable<Partial<Card>> {
    const url = `${environment.apiUrl}/api/v1/regenerate_card_text`;
    
    const request = {
      prompt: card.artPrompt || `Fantasy art of ${card.name}, ${card.type}`,
      width: environment.defaultCardWidth,
      height: environment.defaultCardHeight,
      imageData: imageData, // Include the existing image data
      cardData: {
        name: card.name,
        manaCost: card.manaCost,
        supertype: card.supertype,
        colors: card.colors,
        type: card.type,
        subtype: card.subtype,
        rarity: card.rarity,
        cmc: card.cmc,
        description: card.description,
        power: card.power,
        toughness: card.toughness
      }
    };

    return this.http.post<any>(url, request, { headers: this.defaultHeaders })
      .pipe(
        timeout(30000), // Use shorter timeout for text regeneration (30 seconds)
        map((response: any) => {
          const updatedCardData: Partial<Card> = {};
          
          // Process only card data (ignore any image updates)
          if (response.cardData) {
            try {
              // Try to parse structured card data or use as description
              const parsedData = JSON.parse(response.cardData);
              if (parsedData.name) updatedCardData.name = parsedData.name;
              if (parsedData.description) updatedCardData.description = parsedData.description;
              if (parsedData.flavorText) updatedCardData.flavorText = parsedData.flavorText;
              if (parsedData.manaCost) updatedCardData.manaCost = parsedData.manaCost;
            } catch (e) {
              // If not JSON, treat as description
              updatedCardData.description = response.cardData;
            }
          }
          
          // Process complete card image if returned (with new text overlaid)
          if (response.card_image) {
            updatedCardData.cardImageUrl = `data:image/png;base64,${response.card_image}`;
          }
          
          return updatedCardData;
        }),
        tap(cardData => console.log('Card text regeneration successful:', cardData)),
        catchError(this.handleError<Partial<Card>>('regenerateCardText'))
      );
  }

  /**
   * Generate only card text content (no image) - Legacy method using full endpoint
   * @param card Card data for text generation
   * @returns Observable with updated card text
   */
  generateCardText(card: Card): Observable<Partial<Card>> {
    const url = `${environment.apiUrl}${environment.cardGenerationEndpoint}`;
    
    const request: CardGenerationRequest = {
      prompt: card.artPrompt || `Fantasy art of ${card.name}, ${card.type}`,
      width: environment.defaultCardWidth,
      height: environment.defaultCardHeight,
      cardData: {
        name: card.name,
        manaCost: card.manaCost,  // Added missing mana cost field
        supertype: card.supertype,
        colors: card.colors,
        type: card.type,
        subtype: card.subtype,
        rarity: card.rarity,
        cmc: card.cmc,
        description: card.description,
        power: card.power,
        toughness: card.toughness
      }
    };

    return this.http.post<CardGenerationResponse>(url, request, { headers: this.defaultHeaders })
      .pipe(
        timeout(this.getDynamicTimeout()),
        map((response: CardGenerationResponse) => {
          const updatedCardData: Partial<Card> = {};
          
          // Process only card data (ignore image)
          if (response.cardData) {
            try {
              // Try to parse structured card data or use as description
              const parsedData = JSON.parse(response.cardData);
              if (parsedData.name) updatedCardData.name = parsedData.name;
              if (parsedData.description) updatedCardData.description = parsedData.description;
              if (parsedData.flavorText) updatedCardData.flavorText = parsedData.flavorText;
              if (parsedData.manaCost) updatedCardData.manaCost = parsedData.manaCost;
            } catch (e) {
              // If not JSON, treat as description
              updatedCardData.description = response.cardData;
            }
          }
          
          return updatedCardData;
        }),
        tap(cardData => console.log('Card text generation successful:', cardData)),
        catchError(this.handleError<Partial<Card>>('generateCardText'))
      );
  }

  /**
   * Generate only card artwork (no text content)
   * @param card Card data for art generation
   * @returns Observable with updated card image
   */
  generateCardArt(card: Card): Observable<Partial<Card>> {
    const url = `${environment.apiUrl}${environment.cardGenerationEndpoint}`;
    
    const request: CardGenerationRequest = {
      prompt: card.artPrompt || `Fantasy art of ${card.name}, ${card.type}`,
      width: environment.defaultCardWidth,
      height: environment.defaultCardHeight,
      cardData: {
        name: card.name,
        manaCost: card.manaCost,  // Added missing mana cost field
        supertype: card.supertype,
        colors: card.colors,
        type: card.type,
        subtype: card.subtype,
        rarity: card.rarity,
        cmc: card.cmc,
        description: card.description,
        power: card.power,
        toughness: card.toughness
      }
    };

    return this.http.post<CardGenerationResponse>(url, request, { headers: this.defaultHeaders })
      .pipe(
        timeout(this.getDynamicTimeout()),
        map((response: any) => {
          const updatedCardData: Partial<Card> = {};
          
          // Process only image data (ignore card content)
          if (response.imageData) {
            updatedCardData.imageUrl = `data:image/png;base64,${response.imageData}`;
          }
          
          return updatedCardData;
        }),
        tap(cardData => console.log('Card art generation successful:', cardData)),
        catchError(this.handleError<Partial<Card>>('generateCardArt'))
      );
  }

  /**
   * Save card to user's collection (to be implemented)
   * @param card Card to save
   * @returns Observable with save status
   */
  saveCard(card: Card): Observable<{success: boolean, id: string}> {
    // This would connect to a backend API to save the card
    // For now, we'll just simulate success
    console.log('Saving card:', card);
    return of({ success: true, id: this.generateRandomId() });
  }

  /**
   * Generate a random ID (for demo purposes)
   * @returns Random string ID
   */
  private generateRandomId(): string {
    return Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15);
  }

  /**
   * Enhanced error handler for Lambda proxy requests
   * @param operation Name of the operation that failed
   * @returns Error handler function
   */
  private handleError<T>(operation = 'operation') {
    return (error: HttpErrorResponse): Observable<never> => {
      // Log error to console
      console.error(`${operation} failed:`, error);

      // Check for timeout errors from the RxJS timeout operator
      if ((error as any).name === 'TimeoutError' || error.message?.includes('Timeout')) {
        console.log('⏰ Timeout detected - showing queue toast');
        this.snackBar.open(
          'The queue is large and generation is taking longer than expected. Please try again in a minute.',
          'Dismiss',
          {
            duration: 8000, // Show for 8 seconds
            verticalPosition: 'bottom',
            horizontalPosition: 'center',
            panelClass: ['timeout-toast']
          }
        );
        return throwError(() => new Error('Request timed out. The queue is large - please try again in a minute.'));
      }

      let errorMessage = 'An unknown error occurred';
      
      if (error.error instanceof ErrorEvent) {
        // Client-side error
        errorMessage = `Network error: ${error.error.message}`;
      } else {
        // Server-side error - handle Lambda proxy specific responses
        try {
          const apiError = error.error as any;
          
          // Check for Lambda proxy error responses
          if (apiError.error && apiError.details) {
            if (error.status === 504) {
              errorMessage = 'Request timeout - the card generation is taking longer than expected. Please try again.';
            } else if (error.status === 502) {
              errorMessage = `Proxy error: ${apiError.details}`;
            } else {
              errorMessage = apiError.error;
            }
          } else if (apiError.message) {
            errorMessage = apiError.message;
          } else {
            // Standard HTTP error
            switch (error.status) {
              case 0:
                errorMessage = 'Unable to connect to server. Please check your internet connection.';
                break;
              case 504:
                errorMessage = 'Request timeout. The server is taking too long to respond.';
                break;
              case 502:
                errorMessage = 'Bad Gateway. The proxy server received an invalid response.';
                break;
              case 500:
                errorMessage = 'Internal server error. Please try again later.';
                break;
              default:
                errorMessage = `Server error: ${error.status} ${error.statusText}`;
            }
          }
        } catch (e) {
          errorMessage = `Server error: ${error.status} ${error.statusText}`;
        }
      }

      // Return an observable with a user-facing error message
      return throwError(() => new Error(`${operation} failed: ${errorMessage}`));
    };
  }
}

