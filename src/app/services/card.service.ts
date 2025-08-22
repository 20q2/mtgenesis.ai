import { Injectable } from '@angular/core';
import { HttpClient, HttpErrorResponse, HttpHeaders } from '@angular/common/http';
import { Observable, throwError, of } from 'rxjs';
import { catchError, retry, map, tap, switchMap } from 'rxjs/operators';
import { environment } from '../../environments/environment';
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
   * API URL from environment configuration
   */
  private readonly apiUrl = environment.apiUrl;
  
  /**
   * Number of retry attempts for API calls
   */
  private readonly retryCount = environment.retryAttempts;
  
  /**
   * Delay between retry attempts in milliseconds
   */
  private readonly retryDelay = environment.retryDelay;

  /**
   * Default HTTP headers
   */
  private readonly defaultHeaders = new HttpHeaders({
    'Content-Type': 'application/json'
  });

  constructor(private http: HttpClient) { }



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
        retry({ count: this.retryCount, delay: this.retryDelay }),
        map(response => {
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
          
          return updatedCard;
        }),
        tap(card => console.log('Unified card generation successful:', card)),
        catchError(this.handleError<Card>('generateCompleteCard'))
      );
  }

  /**
   * Generate only card text content (no image)
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
        retry({ count: this.retryCount, delay: this.retryDelay }),
        map(response => {
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
        retry({ count: this.retryCount, delay: this.retryDelay }),
        map(response => {
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
   * Error handler for HTTP requests
   * @param operation Name of the operation that failed
   * @returns Error handler function
   */
  private handleError<T>(operation = 'operation') {
    return (error: HttpErrorResponse): Observable<never> => {
      // Log error to console
      console.error(`${operation} failed:`, error);

      let errorMessage = 'An unknown error occurred';
      
      if (error.error instanceof ErrorEvent) {
        // Client-side error
        errorMessage = `Client error: ${error.error.message}`;
      } else {
        // Server-side error
        try {
          const apiError = error.error as ApiErrorResponse;
          errorMessage = apiError.message || `Server error: ${error.status} ${error.statusText}`;
        } catch (e) {
          errorMessage = `Server error: ${error.status} ${error.statusText}`;
        }
      }

      // Return an observable with a user-facing error message
      return throwError(() => new Error(`${operation} failed: ${errorMessage}`));
    };
  }
}

